import os
import argparse
import numpy as np
import scipy.linalg as sla

#data_dir = '/global/cfs/cdirs/desi/users/mpinon/'
data_dir = '/Users/mp270220/Work/fiber_collisions/'


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


def get_power_marg_likelihood(data, theory, wmatrix, cov, shotnoise=None, ells=[0, 2, 4], idces=[-1]):

    d = data.copy()
    th = theory.deepcopy()

    if shotnoise is not None:
        d[0] += shotnoise
        th.power[0] += shotnoise

    d = d.flatten()

    invcov = np.linalg.inv(cov)
 
    anew = aprime(invcov, wmatrix, idces, ells)
    lda, m = sla.eig(anew)
    mk = m.copy()
    lda[lda < 1e-15] = 0
    mk[:, lda==0] = 0

    wnew = wmatrix.deepcopy()
    wnew.value = (m.dot(mk.T).dot(wmatrix.value.T)).T
    dnew = m.dot(mk.T).dot(d)

    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    
    klim={0: [0.02, 0.2, 0.005], 2: [0.02, 0.2, 0.005], 4: [0.02, 0.2, 0.005]}
    observable = TracerPowerSpectrumMultipolesObservable(klim=klim,
                                                         data=dnew,
                                                         wmatrix=wnew,
                                                         theory=th)
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov)
    return likelihood
