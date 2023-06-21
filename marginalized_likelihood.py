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


data_dir = '/global/cfs/cdirs/desi/users/mpinon/'
#data_dir = '/Users/mp270220/Work/fiber_collisions/'


def get_t(wmatrix, idces, ells):
    nells = len(ells)

    idces = list(idces)

    t = np.zeros((wmatrix.shape[0], nells))
    for idx in idces:
        for i in range(nells):
            t[i * (wmatrix.shape[0]  // nells) + idx][i] = 1
    return t


def aprime(a, w, idces, ells):
    t = get_t(w, idces, ells)
    w = w.value.T
    tmp = (w.dot(t)).T.dot(a).dot(w.dot(t))
    tmpinv = np.linalg.inv(tmp)
    tmp2 = t.dot(tmpinv).dot(t.T)
    anew= (a @ w.dot(tmp2).dot(w.T) @ a)
    return a - anew


def get_power_marg_likelihood(data, theory, wmatrix, cov, shotnoise=None, ells=[0, 2, 4], idces=[-1], solve=True):

    d = data.copy()
    th = theory.deepcopy()

    if shotnoise is not None: d[0] += shotnoise
    d = d.flatten()

    invcov = np.linalg.inv(cov)
 
    anew = aprime(invcov, wmatrix, idces, ells)
    lda, m = sla.eig(anew)
    m = m.real
    mk = m.copy()
    lda[lda < 1e-15] = 0
    mk[:, lda==0] = 0

    wnew = wmatrix.deepcopy()
    wnew.value = (m.dot(mk.T).dot(wmatrix.value.T)).T
    dnew = m.dot(mk.T).dot(d)
    dnew[:len(data[0])] -= shotnoise

    klim={0: [0.02, 0.2, 0.005], 2: [0.02, 0.2, 0.005], 4: [0.02, 0.2, 0.005]}
    observable = TracerPowerSpectrumMultipolesObservable(klim=klim,
                                                         data=dnew,
                                                         wmatrix=wnew,
                                                         theory=th,
                                                         shotnoise=shotnoise)
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov)
    
    likelihood.all_params['b1'].update(ref={'limits': [0.25, 0.35]})
    if solve:
        for param in likelihood.all_params.select(name=['alpha*', 'sn*', 'c*']): param.update(derived='.auto')
        theory.log_info('Use analytic marginalization for {}.'.format(theory.params.names(solved=True)))
    for param in likelihood.all_params.select(basename=['alpha6']):
        param.update(fixed=True)

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
    parser.add_argument('--direct', type=bool, required=False, default=True)
    args = parser.parse_args()

    tracer = args.tracer
    region = args.region
    completeness = args.completeness
    theory_name = args.theory_name
    todo = args.todo
    fc = args.fc
    rp_cut = args.rp_cut
    direct = args.direct
    
    emulator_dir = os.path.join(data_dir, 'emulators/emulators_shapefit_{}'.format(tracer))
    profiles_dir = os.path.join(data_dir, 'profiles/profiles_shapefit_{}_{}{}'.format(tracer, completeness, region))
    chains_dir = os.path.join(data_dir, 'chains/chains_shapefit_{}_{}{}'.format(tracer, completeness, region))
    
    emulator_fn = os.path.join(emulator_dir, 'power_xinmax0.35_{}.npy'.format(theory_name))
    footprint_fn = os.path.join(emulator_dir, 'footprint_{}.npy')
    
    data = np.load(os.path.join(data_dir, 'power_spectra/power_spectrum_25mocks_{}_{}{}{}{}.npy'.format(tracer, completeness, region, '_zcut' if completeness else '', '_th{:.1f}'.format(rp_cut) if rp_cut else '')))
    #theory = LPTVelocileptorsTracerPowerSpectrumMultipoles.load(os.path.join(data_dir, 'theory_{}.npy'.format(tracer)))
    from desilike.emulators import EmulatedCalculator
    pt = EmulatedCalculator.load(emulator_fn)
    theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(pt=pt)
    wmatrix = BaseMatrix.load(os.path.join(data_dir, 'windows/wm_mock0_{}_{}{}{}_rebinned.npy'.format(tracer, completeness, region, '_rp{:.1f}'.format(rp_cut) if rp_cut else '')))
    cov = np.load(os.path.join(data_dir, 'cov_{}_{}{}{}.npy'.format(tracer, completeness, region, '_rp{:.1f}'.format(rp_cut) if rp_cut else '')))
    shotnoise = np.load(os.path.join(data_dir, 'shotnoise_pk_25mocks_{}_{}{}{}.npy'.format(tracer, completeness, region, '_th{:.1f}'.format(rp_cut) if rp_cut else '')))
                    
    print('likelihood')
    likelihood = get_power_marg_likelihood(data, theory, wmatrix, cov, shotnoise=shotnoise, ells=[0, 2, 4], idces=[-1], solve=True)
 
    if 'profiling' in todo:
        print('profiling')
        profiler = MinuitProfiler(likelihood, seed=43, save_fn=os.path.join(profiles_dir, 'power_xinmax0.35_{}{}{}_marginalized.npy'.format(theory_name, fc, '_th{:.1f}'.format(rp_cut) if rp_cut else '')))
        profiler.maximize(niterations=10)
        #print(profiler.profiles.to_stats(tablefmt='pretty'))
    
    if 'sampling' in todo:
        print('sampling')
        sampler = EmceeSampler(likelihood, chains=8, nwalkers=40, seed=43, save_fn=os.path.join(chains_dir, 'power_xinmax0.35_{}{}{}_*_marginalized.npy'.format(theory_name, fc, '_th{:.1f}'.format(rp_cut) if rp_cut else '')))
        sampler.run(check={'max_eigen_gr': 0.02})