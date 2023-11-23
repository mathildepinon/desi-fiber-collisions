import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(os.path.join(os.path.abspath(''), 'plot_style.mplstyle'))
plt.figure(figsize=(15,10))

from pycorr import TwoPointCorrelationFunction, TwoPointEstimator, LandySzalayTwoPointEstimator, NaturalTwoPointEstimator,\
                   project_to_multipoles, project_to_wp, setup_logging

from pycorr.twopoint_estimator import project_to_wedges


class mock:
    
    def __init__(self, mock_id, footprint, tracer, tr=1, rebin=(1, 1), rp_threshold=0, theta_threshold=0):
        ext=".npy"
        special = ""
        #xi_dir = "/global/cfs/cdirs/desi/users/eburtin/"
        #xi_dir = '/global/cfs/cdirs/desi/users/mpinon/correlation_functions/'
        xi_dir = '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/xi/'
        xi_fn = 'corr_func_mock{:d}_{}_{{}}{}'.format(mock_id, tracer, footprint)
#        self.xi_rppi = TwoPointCorrelationFunction.load("fa_mocks/ELG_"+footprint+"rppi_"+special+"mock_"+str(mock_id)+ext)
        self.xi_smu = TwoPointCorrelationFunction.load(xi_dir+xi_fn.format('')+ext)[::rebin[0],::rebin[1]]
#        self.xi_complete_rppi = TwoPointCorrelationFunction.load("fa_mocks/ELG_"+footprint+"rppi_c_"+special+"mock_"+str(mock_id)+ext)
        self.xi_complete_smu = TwoPointCorrelationFunction.load(xi_dir+xi_fn.format('complete_')+ext)[::rebin[0],::rebin[1]]
        self.xi_mp = project_to_multipoles(self.xi_smu)
        self.xi_complete_mp = project_to_multipoles(self.xi_complete_smu)
        self.xi_mp_tr = project_to_multipoles(self.xi_smu[:, tr:-tr])
        self.xi_complete_mp_tr = project_to_multipoles(self.xi_complete_smu[:, tr:-tr])
        
        # rp-cut
        self.rp_threshold = rp_threshold
        self.theta_threshold = theta_threshold
        self.xi_rpcut = TwoPointCorrelationFunction.load(os.path.join(xi_dir, xi_fn.format('')+('_rpcut{:.1f}'.format(rp_threshold) if rp_threshold else '')+ext))[::rebin[0],::rebin[1]]
        self.xi_complete_rpcut = TwoPointCorrelationFunction.load(os.path.join(xi_dir, xi_fn.format('complete_')+('_rpcut{:.1f}'.format(rp_threshold) if rp_threshold else '')+ext))[::rebin[0],::rebin[1]]
        self.xi_thetacut = TwoPointCorrelationFunction.load(os.path.join(xi_dir, xi_fn.format('')+('_thetacut{:.2f}'.format(theta_threshold) if theta_threshold else '')+ext))[::rebin[0],::rebin[1]]
        self.xi_complete_thetacut = TwoPointCorrelationFunction.load(os.path.join(xi_dir, xi_fn.format('complete_')+('_thetacut{:.2f}'.format(theta_threshold) if theta_threshold else '')+ext))[::rebin[0],::rebin[1]]
        self.xi_mp_rpcut = project_to_multipoles(self.xi_rpcut, ignore_nan=True)
        self.xi_complete_mp_rpcut = project_to_multipoles(self.xi_complete_rpcut, ignore_nan=True)
        self.xi_mp_thetacut = project_to_multipoles(self.xi_thetacut, ignore_nan=True)
        self.xi_complete_mp_thetacut = project_to_multipoles(self.xi_complete_thetacut, ignore_nan=True)
        # Apply rp-cut when binning in (s, mu)
        self.xi_mp_smu_rpcut = self.xi_smu[:, 1:-1](ells=(0, 2, 4), rp=(rp_threshold, np.inf), return_sep=True, ignore_nan=True)
        self.xi_complete_mp_smu_rpcut = self.xi_complete_smu[:, 1:-1](ells=(0, 2, 4), rp=(rp_threshold, np.inf), return_sep=True, ignore_nan=True)

   
    def plot(self):
        color=['dodgerblue', 'orangered', 'darkcyan', 'firebrick', 'violet', 'olivedrab', 'gold', 'limegreen', 'darkorange', 'darkviolet', 'deepskyblue']
        power=2
        for mp in range(3):
            plt.plot(self.xi_mp[0], self.xi_mp[1][mp]*self.xi_mp[0]**power, linestyle=':', label=r'$\ell = {:d}$ FA'.format(2*mp), color=color[mp])
            plt.plot(self.xi_complete_mp[0], self.xi_complete_mp[1][mp]*self.xi_complete_mp[0]**power, label=r'$\ell = {:d}$ complete'.format(2*mp), color=color[mp])
            

def get_average_multipole(mocks, iell):
    nmocks = len(mocks)
    names = ['xi_mp', 'xi_complete_mp', 'xi_mp_tr', 'xi_complete_mp_tr', 'xi_mp_rpcut', 'xi_complete_mp_rpcut', 'xi_mp_thetacut', 'xi_complete_mp_thetacut', 'xi_mp_smu_rpcut', 'xi_complete_mp_smu_rpcut']
    res = dict()
    for name in names:
        mocks_xis = np.array([getattr(mocks[i], name)[1][iell] for i in range(nmocks)])
        xi_mean = np.mean(mocks_xis, axis=0)
        xi_std = np.std(mocks_xis, axis=0)
        res[name] = (xi_mean, xi_std)
    return(res)


def plot_fiber_collisions(mocks, mp, max_vals=None):
    names = ['xi_mp', 'xi_complete_mp', 'xi_mp_tr', 'xi_complete_mp_tr', 'xi_mp_rpcut', 'xi_complete_mp_rpcut', 'xi_mp_thetacut', 'xi_complete_mp_thetacut', 'xi_mp_smu_rpcut', 'xi_complete_mp_smu_rpcut']
    rp_threshold = mocks[0].rp_threshold
    theta_threshold = mocks[0].theta_threshold
    labels = {'xi_mp': 'FA', 'xi_complete_mp': 'Complete', 
              'xi_mp_tr': r'FA, $\mu$-truncated', 'xi_complete_mp_tr': r'Complete, $\mu$-truncated',
              'xi_mp_rpcut': r'FA, cutting $r_p < {} \; \mathrm{{Mpc}}/h$'.format(rp_threshold), 'xi_complete_mp_rpcut': r'Cutting $r_{{\perp}} < {} \; \mathrm{{Mpc}}/h$'.format(rp_threshold),
              'xi_mp_thetacut': r'FA, cutting $\theta < {}°$'.format(theta_threshold), 'xi_complete_mp_thetacut': r'Cutting $\theta < {}°$'.format(theta_threshold),
              'xi_mp_smu_rpcut': r'FA, (s, $\mu$) binning $r_p$-cut'.format(rp_threshold), 'xi_complete_mp_smu_rpcut': r'Complete,  (s, $\mu$) binning $r_p$-cut'.format(rp_threshold)}
    linestyles = {'xi_mp': '--', 'xi_complete_mp': '-', 'xi_mp_tr': ':', 'xi_complete_mp_tr': '--', 'xi_mp_rpcut': '--', 'xi_complete_mp_rpcut': '-', 'xi_mp_thetacut': '--', 'xi_complete_mp_thetacut': '-', 'xi_mp_smu_rpcut': ':', 'xi_complete_mp_smu_rpcut': '-'}
    alphas = {'xi_mp': 1., 'xi_complete_mp': 1., 'xi_mp_tr': 0.2, 'xi_complete_mp_tr': 0.2, 'xi_mp_rpcut': 0.8, 'xi_complete_mp_rpcut': 0.8, 'xi_mp_thetacut': 0.4, 'xi_complete_mp_thetacut': 0.4, 'xi_mp_smu_rpcut': 0.2, 'xi_complete_mp_smu_rpcut': 0.2}

    power = 2
    r = mocks[0].xi_mp[0]
    
    fig,axs = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    color=['dodgerblue', 'orangered', 'darkcyan', 'firebrick', 'violet', 'olivedrab', 'gold', 'limegreen', 'darkorange', 'darkviolet', 'deepskyblue']

    for i, ell in enumerate(mp):
        avg_mp = get_average_multipole(mocks, i)
        for name in ['xi_complete_mp', 'xi_mp', 'xi_complete_mp_rpcut', 'xi_mp_rpcut', 'xi_complete_mp_thetacut', 'xi_mp_thetacut']:
            res, std = avg_mp[name]
            lab = labels[name] if (('complete' in name) or (not 'cut' in name)) else ''
            if 'cut' in name:
                axs[i].plot(r, r**power*res, color=color[i], ls=linestyles[name], alpha=alphas[name])
            axs[i].plot([], [], label=lab, color='black', ls=linestyles[name], alpha=alphas[name])

        axs[i].set_xlabel('$r$ [Mpc/$h$]')
        axs[i].set_title(r'$\ell={:d}$'.format(ell))
    axs[0].set_ylabel(r'$r^2 \xi_\ell$ [$(\mathrm{Mpc}/h)^{2}$]')
    axs[0].legend()
    fig.tight_layout(pad=0.3)
    
    
def plot_delta_fiber_collisions(mocks, mp, max_vals=None):
    names = ['xi_mp', 'xi_mp_tr', 'xi_mp_rpcut']
    rp_threshold = mocks[0].rp_threshold
    labels = {'xi_mp': 'No cut', 'xi_mp_tr': r'FA - complete, $\mu$-truncated', 'xi_mp_rpcut':  'Cutting $r_p < {} \; \mathrm{{Mpc}}/h$'.format(rp_threshold)}
    linestyles = {'xi_mp': '-', 'xi_mp_tr': ':', 'xi_mp_rpcut': '--'}

    power = 2
    r = mocks[0].xi_mp[0]
    
    fig,axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    color=['dodgerblue', 'orangered', 'darkcyan', 'firebrick', 'violet', 'olivedrab', 'gold', 'limegreen', 'darkorange', 'darkviolet', 'deepskyblue']

    for i, ell in enumerate(mp):
        avg_mp = get_average_multipole(mocks, i)
        for tr in ['', '_rpcut']:
            name = 'xi_mp{}'.format(tr)
            name_complete = 'xi_complete_mp{}'.format(tr)
            res, std = avg_mp[name]
            res_c, std_c = avg_mp[name_complete]
            lab = labels[name]
            axs[i].plot(r, r**power*(res - res_c), color=color[i], ls=linestyles[name])
            axs[i].plot([], [], label=lab, color='black', ls=linestyles[name])
            if name == 'xi_mp_rpcut':
                axs[i].fill_between(r, -std*r**power/np.sqrt(len(mocks)), std*r**power/np.sqrt(len(mocks)),label=r'$\sigma_\mathrm{{Y1}}/{:.0f}$'.format(np.sqrt(len(mocks))), facecolor='grey', alpha=0.1)

        axs[i].set_xlabel('$r$ [Mpc/$h$]')
        axs[i].set_title(r'$\ell={:d}$'.format(ell))
    axs[0].set_ylabel(r'$r^2 \Delta \xi_\ell$ [$(\mathrm{Mpc}/h)^{2}$]')
    axs[0].legend()


#mocks = [mock(id,"NS_") for id in range(25)]
#plot_fiber_collisions(mocks,2,6,8)
#eNS = plot_delta_fiber_collisions(mocks,1.5,4,5)

#mocks = [mock(id,"SS_") for id in range(25)]
#plot_fiber_collisions(mocks,2,6,8)
#eSS = plot_delta_fiber_collisions(mocks,2,6,8)
#plt.savefig('fa-mitigate.png')
#mocks = [mock(id,"SS_") for id in range(25)]

#plt.plot(eSS/eNS)