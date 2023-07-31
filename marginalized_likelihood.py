import os
import argparse
import numpy as np
import scipy.linalg as sla

from pypower import BaseMatrix
from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerPowerSpectrumMultipoles
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.profilers import MinuitProfiler
from desilike.samplers import EmceeSampler  
from desilike.emulators import EmulatedCalculator
from emulator_fit import get_power_likelihood

data_dir = '/global/cfs/cdirs/desi/users/mpinon/'
#data_dir = '/Users/mp270220/Work/fiber_collisions/'


def get_likelihood_elements(tracer, region, completeness, stat, theory_name, rp_cut, kobsmax=0.2, xinmax=True, solve=True, fc='', direct=False, save=False, imock=None):
    
    # Window matrix
    #window_fn = os.path.join(data_dir, 'windows/wm_mock0_{}_{}{}{}_rebinned_kobsmax{:.2f}.npy'.format(tracer, completeness, region, '_rp{:.1f}'.format(rp_cut) if rp_cut else '', kobsmax))
    #if not os.path.isfile(window_fn) or save: 
    wm = BaseMatrix.load(os.path.join(data_dir, 'windows/wm_mock0_{}_{}{}{}.npy'.format(tracer, completeness, region, '_rp{:.1f}'.format(rp_cut) if rp_cut else '')))
    w = wm.deepcopy()
    # in
    kinrebin = 10
    w.slice_x(slicein=slice(0, len(w.xin[0]) // kinrebin * kinrebin, kinrebin))
    w.select_x(xinlim=(0.005, 0.35))
    # out
    klim = {0: [0.02, kobsmax, 0.005], 2: [0.02, kobsmax, 0.005], 4: [0.02, kobsmax, 0.005]}
    factorout = 5
    w.slice_x(sliceout=slice(0, len(w.xout[0]) // factorout * factorout, factorout))
    w.select_x(xoutlim=klim[0][:2])    
    #w.save(window_fn)
#wmatrix = BaseMatrix.load(window_fn)
            
    emulator_dir = os.path.join(data_dir, 'emulators/emulators_shapefit_{}'.format(tracer))
    emulator_fn = os.path.join(emulator_dir, 'power{}_{{}}.npy'.format('_xinmax0.35' if xinmax else ''))
    footprint_fn = os.path.join(emulator_dir, 'footprint_{}.npy')
    
    likelihood = get_power_likelihood(tracer=tracer, region=region, completeness=completeness, theory_name=theory_name, fc=fc, rp_cut=rp_cut, direct=direct,
                                      emulator_fn=emulator_fn, footprint_fn=footprint_fn, solve=False, imock=imock, kobsmax=kobsmax)
    
    # Covariance
    #cov_fn = os.path.join(data_dir, 'cov_{}_{}{}.npy'.format(tracer, completeness, region))
#    if not os.path.isfile(cov_fn) or save: 
#        cov = likelihood.observables[0].covariance
#        np.save(cov_fn, cov)
#    cov = np.load(cov_fn)
    cov = likelihood.observables[0].covariance
        
    # Theory
    pt = EmulatedCalculator.load(emulator_fn.format(theory_name))
    theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(pt=pt)

    # Theory vector
    theory_vector = np.array(likelihood.observables[0].wmatrix.theory.power).flatten()

    # Data vector
    #data_fn = os.path.join(data_dir, 'power_spectra/power_spectrum_25mocks_{}_{}{}{}{}.npy'.format(tracer, completeness, region, '_zcut' if completeness else '', '_th{:.1f}'.format(rp_cut) if rp_cut else ''))
    #if not os.path.isfile(data_fn) or save: 
    #    np.save(data_fn, likelihood.observables[0].data)
    #data = np.load(data_fn, allow_pickle=True)
    data = np.array(likelihood.observables[0].data)
                
    # Shot noise
    #shotnoise_fn = os.path.join(data_dir, 'shotnoise_pk_25mocks_{}_{}{}{}.npy'.format(tracer, completeness, region, '_th{:.1f}'.format(rp_cut) if rp_cut else ''))
#    if not os.path.isfile(shotnoise_fn) or save: 
#        np.save(shotnoise_fn, likelihood.observables[0].shotnoise.flat[0])
#    shotnoise = np.load(shotnoise_fn, allow_pickle=True)
    shotnoise = likelihood.observables[0].shotnoise.flat[0]
    #np.save(shotnoise_fn, likelihood.observables[0].shotnoise.flat[0])
        
    return {'wmatrix': w, 'covariance': cov, 'theory': theory, 'theory_vector': theory_vector, 'data': data, 'shotnoise': shotnoise}


def get_t(wmatrix, idces, ells):
    nells = len(ells)

    idces = list(idces)

    t = np.zeros((wmatrix.shape[0], nells))
    for idx in idces:
        for i in range(nells):
            t[i * (wmatrix.shape[0]  // nells) + idx][i] = 1
    return t


def aprime(a, w, idces, ells, c=None, prior=False):
    t = get_t(w, idces, ells)
    w = w.value.T
    h = w.dot(t)

    if prior:
        if c is None:
            c = np.diag(np.full(len(ells), prior))

        b = np.linalg.inv(h.T.dot(a).dot(h) + c).dot(h.T).dot(a)
        correction = 2 * b.T.dot(h.T).dot(a) - b.T.dot(h.T).dot(a).dot(h).dot(b)
        anew = a - correction
    else:
        tmp = (w.dot(t)).T.dot(a).dot(w.dot(t))
        tmpinv = np.linalg.inv(tmp)
        tmp2 = t.dot(tmpinv).dot(t.T)
        anew = a - (a @ w.dot(tmp2).dot(w.T) @ a)
    return anew


def get_new_elements(data, wmatrix, cov, wmatrix_rpcut=None, shotnoise=None, ells=[0, 2, 4], idces=[-1]):
    d = data.copy()

    if shotnoise is not None: d[0] += shotnoise
    d = d.flatten()
    
    invcov = np.linalg.inv(cov)

    # when computing the likelihood for data without rp-cut, provide the window matrix with rp-cut to compute the rotation matrix m
    if wmatrix_rpcut is not None:
        anew = aprime(invcov, wmatrix_rpcut, idces, ells)
    else:
        anew = aprime(invcov, wmatrix, idces, ells)
    lda, m = sla.eigh(anew)
    #m = m.real
    mk = m.copy()
    lda[lda < sorted(lda)[3]] = 0
    mk[:, lda==0] = 0

    wnew = wmatrix.deepcopy()

    if wmatrix_rpcut is None:
        wnew.value = (m.dot(mk.T).dot(wmatrix.value.T)).T
        dnew = m.dot(mk.T).dot(d)
        dnew[:len(data[0])] -= shotnoise
        
        idx = np.where(lda==0)[0]
        lsub = m.T.dot(invcov).dot(m)[idx, idx]
        lda_new = lda.copy()
        lda_new[lda==0] = lsub
        astar = m.dot(np.diag(lda_new)).dot(m.T)
        covnew = np.linalg.inv(astar)
    
    else:
        m_trunc = np.delete(m, lda==0, axis=1)
        mk_trunc = np.delete(mk, lda==0, axis=1)
        print(mk_trunc.shape)
        mc = m_trunc.dot(mk_trunc.T)
        covnew = mk_trunc.T.dot(cov).dot(mk_trunc)
        dnew = mk_trunc.T.dot(d)
        wnew.value = (mk_trunc.T.dot(wmatrix.value.T)).T
        wnew.xout = [xout[:-1] for xout in wnew.xout]
        dnew[:len(wnew.xout[0])] -= shotnoise
    
    return {'wmatrix': wnew, 'covariance': covnew, 'data': dnew, 'M': m, 'Lambda': lda}


def get_power_marg_likelihood(data, theory, wmatrix, cov, wmatrix_rpcut=None, shotnoise=None, ells=[0, 2, 4], idces=[-1], solve=True):

    likelihood_elements = get_new_elements(data, wmatrix, cov, wmatrix_rpcut, shotnoise, ells, idces)
    data = likelihood_elements['data']
    cov = likelihood_elements['covariance']
    wmatrix = likelihood_elements['wmatrix']

    if wmatrix_rpcut is None:
        klim = {0: [0.02, 0.2, 0.005], 2: [0.02, 0.2, 0.005], 4: [0.02, 0.2, 0.005]}
    else:
        klim = {0: [0.02, 0.195, 0.005], 2: [0.02, 0.195, 0.005], 4: [0.02, 0.195, 0.005]}
    
    observable = TracerPowerSpectrumMultipolesObservable(klim=klim,
                                                         data=data,
                                                         wmatrix=wmatrix,
                                                         theory=theory,
                                                         shotnoise=shotnoise)
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov)
    
    likelihood.all_params['b1'].update(ref={'limits': [0.25, 0.35]})
    if solve:
        for param in likelihood.all_params.select(name=['alpha*', 'sn*', 'c*']): param.update(derived='.auto')
        theory.log_info('Use analytic marginalization for {}.'.format(theory.params.names(solved=True)))
    for param in likelihood.all_params.select(basename=['alpha6']):
        param.update(fixed=True)
    
    #likelihood()
    #likelihood.precision = anew

    return likelihood


if __name__ == '__main__':
    
    from desilike import setup_logging
    
    setup_logging()
        
    parser = argparse.ArgumentParser(description='Emulator fit')
    parser.add_argument('--tracer', type=str, required=False, default='ELG', choices=['ELG', 'LRG', 'QSO'])
    parser.add_argument('--region', type=str, required=False, default='SGC', choices=['NGC', 'SGC', 'NS', 'SS'])
    parser.add_argument('--completeness', type=str, required=False, default='', choices=['', 'complete_'])
    parser.add_argument('--todo', type=str, required=False, default='emulator', choices=['emulator', 'profiling', 'sampling', 'importance'])
    parser.add_argument('--theory_name', type=str, required=False, default='velocileptors', choices=['pybird', 'velocileptors'])
    parser.add_argument('--fc', type=str, required=False, default='', choices=['', '_fc'])
    parser.add_argument('--rp_cut', type=float, required=False, default=0)
    parser.add_argument('--kobsmax', type=float, required=False, default=0.2)
    parser.add_argument('--marg_idx', nargs='*')
    parser.add_argument('--direct', type=bool, required=False, default=True)
    parser.add_argument('--imock', type=int, required=False, default=None)
    args = parser.parse_args()

    tracer = args.tracer
    region = args.region
    completeness = args.completeness
    theory_name = args.theory_name
    todo = args.todo
    fc = args.fc
    rp_cut = args.rp_cut
    kobsmax = args.kobsmax
    marg_idx = [int(idx) for idx in args.marg_idx] if args.marg_idx is not None else [-1]
    direct = args.direct
    imock = args.imock
    
    emulator_dir = os.path.join(data_dir, 'emulators/emulators_shapefit_{}'.format(tracer))
    profiles_dir = os.path.join(data_dir, 'profiles/profiles_shapefit_{}_{}{}'.format(tracer, completeness, region))
    chains_dir = os.path.join(data_dir, 'chains/chains_shapefit_{}_{}{}'.format(tracer, completeness, region))
    
    stat = 'power'
    likelihood_elements = get_likelihood_elements(tracer, region, completeness, stat, theory_name, rp_cut, kobsmax=kobsmax, imock=imock)
    wmatrix = likelihood_elements['wmatrix']
    cov = likelihood_elements['covariance']
    theory = likelihood_elements['theory']
    data = likelihood_elements['data']
    shotnoise = likelihood_elements['shotnoise']

    if rp_cut:
        wmatrix_rpcut = None
    else:
        wmatrix_rpcut = get_likelihood_elements(tracer, region, completeness, stat, theory_name, 2.5, kobsmax=kobsmax, imock=imock)['wmatrix']
    
    print('likelihood')
    likelihood = get_power_marg_likelihood(data, theory, wmatrix, cov, wmatrix_rpcut=wmatrix_rpcut, shotnoise=shotnoise, ells=[0, 2, 4], idces=[-1], solve=True)
    #likelihood()
    
    if 'profiling' in todo:
        print('profiling')
        profiler = MinuitProfiler(likelihood, seed=43, save_fn=os.path.join(profiles_dir, 'power_xinmax0.35_{}{}{}_marginalized.npy'.format(theory_name, fc, '_th{:.1f}'.format(rp_cut) if rp_cut else '')))
        profiler.maximize(niterations=10)
        #print(profiler.profiles.to_stats(tablefmt='pretty'))
    
    if 'sampling' in todo:
        print('sampling')
        sampler = EmceeSampler(likelihood, chains=8, nwalkers=40, seed=43, save_fn=os.path.join(chains_dir, 'power_xinmax0.35_{}{}{}_*_marginalized_astar_rpcutwmatrix{}.npy'.format(theory_name, fc, '_th{:.1f}'.format(rp_cut) if rp_cut else '',  '_mock{}'.format(imock) if imock is not None else '')))
        sampler.run(check={'max_eigen_gr': 0.02})
        
    if 'importance' in todo:
        from desilike.samplers import ImportanceSampler
        from desilike.samples import Chain
        
        chain = Chain.concatenate([Chain.load(os.path.join(chains_dir, 'power_xinmax0.35_{}_{:d}_marginalized_astar_rpcutwmatrix{}.npy'.format(theory_name, i, '_mock{}'.format(imock) if imock is not None else ''))).remove_burnin(0.5)[::10] for i in range(8)])
        chain.aweight[...] *= np.exp(chain.logposterior.max() - chain.logposterior)

        #save_fn = os.path.join(chains_dir, 'power_importance_xinmax0.35_{}{}{}_marginalized_astar_rpcutwmatrix.npy'.format('_mock{}'.format(imock) if imock is not None else '', theory_name, fc, '_th{:.1f}'.format(rp_cut) if rp_cut else ''))
        save_fn = os.path.join(chains_dir, 'power{}_importance_xinmax0.35_{}{}{}_marginalized_astar_rpcutwmatrix_kobsmax{}.npy'.format('_mock{}'.format(imock) if imock is not None else '', theory_name, fc, '_th{:.1f}'.format(rp_cut) if rp_cut else '', kobsmax))
        sampler = ImportanceSampler(likelihood, chain, save_fn=save_fn)
        sampler.run()
 