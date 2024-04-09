# code adapted from https://github.com/cosmodesi/anotherpipe/tree/main/anotherpipe by Pat McDonald

import os
from matplotlib import pyplot as plt
import numpy as np
try:
    from jax import numpy as jnp
    from jax.config import config; config.update('jax_enable_x64', True)
except ImportError:
    jnp = np


from pypower import utils, BaseMatrix
from pypower.utils import BaseClass, setup_logging
from utils import load_poles
from cov_utils import truncate_cov, get_EZmocks_covariance
from local_file_manager import LocalFileName
from desi_file_manager import DESIFileName


class WindowRotation(BaseClass):

    def __init__(self, wmatrix, covmatrix, **kwargs):
        self.set_wmatrix(wmatrix, **kwargs)
        self.set_covmatrix(covmatrix, **kwargs)

    def set_wmatrix(self, wmatrix, **kwargs):
        self.kin = {proj.ell: x for proj, x in zip(wmatrix.projsin, wmatrix.xin)}
        self.kout = {proj.ell: x for proj, x in zip(wmatrix.projsout, wmatrix.xout)}
        self.wmatrix = wmatrix #np.array(wmatrix.value.T, dtype='f8')

        ellsin = np.concatenate([[ell] * len(x) for ell, x in self.kin.items()])
        self.mask_ellsin = {ell: ellsin == ell for ell in self.kin}
        ellsout = np.concatenate([[ell] * len(x) for ell, x in self.kout.items()])
        self.mask_ellsout = {ell: ellsout == ell for ell in self.kout}

        wm00 = self.wmatrix.value.T[np.ix_(self.mask_ellsout[0], self.mask_ellsin[0])][np.argmin(np.abs(self.kout[0] - 0.1))]  # kout = 0.1 h/Mpc
        height = np.max(wm00)  # peak height
        kmax = self.kin[0][np.argmin(np.abs(wm00 - height))]  # k at maximum
        mask_before = self.kin[0] < kmax
        mask_after = self.kin[0] >= kmax
        k1 = self.kin[0][mask_before][np.argmin(np.abs(wm00[mask_before] - height / 2.))]
        k2 = self.kin[0][mask_after][np.argmin(np.abs(wm00[mask_after] - height / 2.))]
        self.bandwidth = np.abs(k2 - k1) #/ (2. * np.sqrt(2. * np.log(2)))
        self.khalfout = {}
        for ell in self.kout:
            kin = self.kin[ell]
            wm = self.wmatrix.value.T[np.ix_(self.mask_ellsout[ell], self.mask_ellsin[ell])]
            mask_half = (wm / kin) / np.max(wm / kin, axis=1)[:, None] > 0.5
            self.khalfout[ell] = np.sum(wm * mask_half * self.kin[ell], axis=1) / np.sum(wm * mask_half, axis=1)

    def set_covmatrix(self, covmatrix, **kwargs):
        self.covmatrix = np.array(covmatrix)

    def fit(self, Minit='momt', Mtarg=None, max_sigma_W=1000, max_sigma_R=1000, factor_diff_ell=100, csub=False, state=None):
        """Fit."""
        import jax
        import optax

        kin = np.concatenate(list(self.kin.values()))
        kout = np.concatenate(list(self.khalfout.values()))
        ellsin = np.concatenate([[ell] * len(x) for ell, x in self.kin.items()])
        ellsout = np.concatenate([[ell] * len(x) for ell, x in self.kout.items()])

        if Mtarg is None:
            Mtarg = np.eye(len(kout))

        self.csub = csub
        
        if Minit in [None, 'momt']:
            with_momt = Minit == 'momt'
            Minit = jnp.identity(len(kout), dtype=jnp.float32)#np.eye(len(kout))
            offsets = jnp.zeros((len(kout), len(kout)), dtype=jnp.float32)#np.zeros_like(Minit)
            max_offset = 0
            for k in range(1, 1 + max_offset):
                offsets += jnp.identity(len(kout), k=k, dtype=jnp.float32) + jnp.identity(len(kout), k=-k, dtype=jnp.float32)#np.eye(len(kout), k=k) + np.eye(len(kout), k=-k)
            offsets *= ellsout[:, None] * ellsout[...]
            offsets /= np.maximum(np.sum(offsets, axis=1), 1)[:, None]
            Minit -= offsets
            if with_momt:
                mo, mt, m = [], [], []
                kincut = 0.20
                idxout = 20
                for mask_ellout in self.mask_ellsout.values():
                    rowin = self.wmatrix.value.T[mask_ellout, :][idxout, :]
                    mt.append(rowin * (kin >= kincut))
                    mo.append([row[ellsin == ell][-1] / rowin[ellsin == ell][-1] for row, ell in zip(self.wmatrix.value.T, ellsout)] * mask_ellout)
                    m.append(0.)
                #print('mt', np.sum(mt), 'mo', np.sum(mo))
                if csub:
                    Minit = (Minit, mo, mt, m)
                else:
                    Minit = (Minit, mo, mt)
        else:
            with_momt = isinstance(Minit, tuple)

        weights_wmatrix = np.empty_like(self.wmatrix.value.T)
        for io, ko in enumerate(kout):
            weights_wmatrix[io, :] = np.minimum(((kin - ko) / self.bandwidth)**2, max_sigma_W**2)
            weights_wmatrix[io, :] += factor_diff_ell * (ellsout[io] != ellsin)  # off-diagonal blocks
        weights_covmatrix = np.empty_like(self.covmatrix)
        for io, ko in enumerate(kout):
            weights_covmatrix[io, :] = np.minimum(((kout - ko) / self.bandwidth)**2, max_sigma_R**2)
            weights_covmatrix[io, :] += factor_diff_ell * (ellsout[io] != ellsout)  # off-diagonal blocks
        #weights_wmatrix = jax.device_put(weights_wmatrix)
        #weights_covmatrix = jax.device_put(weights_covmatrix)

        def softabs(x):
            return jnp.sqrt(x**2 + 1e-37)
        
        def RfromC(C):
            sig = jnp.sqrt(jnp.diag(C))
            denom = jnp.outer(sig, sig)
            return C / denom

        def loss(mmatrix):
            Wp, Cp = self.rotate(mmatrix=mmatrix)
            if with_momt: mmatrix = mmatrix[0]
            loss_W = jnp.sum(softabs(Wp * weights_wmatrix)) / jnp.sum(softabs(Wp) * (weights_wmatrix > 0))
            Rp = RfromC(Cp)
            loss_C = jnp.sum(softabs(Rp * weights_covmatrix)) / jnp.sum(softabs(Rp) * (weights_covmatrix > 0))
            loss_M = 10 * jnp.sum((jnp.sum(mmatrix, axis=1) - 1.)**2)
            #print(loss_W, loss_C, weights_wmatrix.sum(), weights_covmatrix.sum(), weights_wmatrix.shape, weights_covmatrix.shape)
            return loss_W + loss_C + loss_M

        def fit(theta, loss, init_learning_rate=0.00001, meta_learning_rate=0.0001, nsteps=100000, state=None, meta_state=None):

            self.log_info(f'Will do {nsteps} steps')
            optimizer = optax.inject_hyperparams(optax.adabelief)(learning_rate=init_learning_rate)
            meta_opt = optax.adam(learning_rate=meta_learning_rate)

            @jax.jit
            def step(theta, state):
                grads = jax.grad(loss)(theta)
                updates, state = optimizer.update(grads, state)
                theta = optax.apply_updates(theta, updates)
                return theta, state

            @jax.jit
            def outer_loss(eta, theta, state):
                # Apparently this is what inject_hyperparams allows us to do
                state.hyperparams['learning_rate'] = jnp.exp(eta)
                theta, state = step(theta, state)
                return loss(theta), (theta, state)

            # Only this jit actually matters
            @jax.jit
            def outer_step(eta, theta, meta_state, state):
                #has_aux says we're going to return the 2nd part, extra info
                grad, (theta, state) = jax.grad(outer_loss, has_aux=True)(eta, theta, state)
                meta_updates, meta_state = meta_opt.update(grad, meta_state)
                eta = optax.apply_updates(eta, meta_updates)
                return eta, theta, meta_state, state

            if state is None: state = optimizer.init(theta)
            eta = jnp.log(init_learning_rate)
            if meta_state is None: meta_state = meta_opt.init(eta)
            printstep = max(nsteps // 20, 1)
            self.log_info(f'Initial loss: {loss(theta)}')
            for i in range(nsteps):
                eta, theta, meta_state, state = outer_step(eta, theta, meta_state, state)
                if i < 2 or nsteps - i < 4 or i % printstep == 0:
                    self.log_info(f'step {i}, loss: {loss(theta)}, lr: {jnp.exp(eta)}')
            return theta, (jnp.exp(eta), meta_state, state)

        if state is None:
            self.mmatrix, self.state = fit(Minit, loss)
        else:
            self.mmatrix, self.state = fit(Minit, loss, init_learning_rate=state[0], state=state[2], meta_state=state[1])
        return self.mmatrix, self.state

    def rotate(self, mmatrix=None, obs=None, theory=None):
        """Return prior and precmatrix if input theory."""
        if mmatrix is None: mmatrix = self.mmatrix
        with_momt = isinstance(mmatrix, tuple)
        if not hasattr(self, 'csub'):
            self.csub = len(mmatrix)>3
        if with_momt:      
            Wsub = jnp.zeros(self.wmatrix.value.T.shape)
            if self.csub:
                mmatrix, mo, mt, m = mmatrix
                Csub = jnp.zeros(self.covmatrix.shape)
                for mmo, mmt, mm, mask_ellout in zip(mo, mt, m, self.mask_ellsout.values()):
                    mask_mo = mask_ellout * mmo
                    Wsub += jnp.outer(mask_mo, mmt)
                    Csub += mm * jnp.outer(mask_mo, mask_mo)
            else:
                mmatrix, mo, mt = mmatrix
                Csub = 0
                for mmo, mmt, mask_ellout in zip(mo, mt, self.mask_ellsout.values()):
                    mask_mo = mask_ellout * mmo
                    Wsub += jnp.outer(mask_mo, mmt)
        else:
            Wsub = Csub = 0.
        #print('WC', Wsub.sum(), Csub.sum())
        Wp = jnp.matmul(mmatrix, self.wmatrix.value.T) - Wsub
        Cp = jnp.matmul(jnp.matmul(mmatrix, self.covmatrix), mmatrix.T) - Csub
        if obs is not None:
            obs = np.asarray(obs, dtype='f8').ravel()
            Pp = np.matmul(mmatrix, obs)
            if theory is not None and with_momt:
                theory = np.asarray(theory, dtype='f8').ravel()
                precmatrix = np.linalg.inv(Cp)
                deriv = np.array(mo)
                fisher = deriv.dot(precmatrix).dot(deriv.T)
                derivp = deriv.dot(precmatrix)
                m = np.linalg.solve(fisher, derivp.dot(Pp - np.matmul(Wp, theory)))
                fisher += np.diag(1. / m**2)  # prior
                precmatrix = precmatrix - derivp.T.dot(np.linalg.solve(fisher, derivp))
                offset = np.dot(m, mo)
                return Wp, Cp, Pp, m, precmatrix, offset
            return Wp, Cp, Pp
        return Wp, Cp

    def plot_wmatrix(self, k=0.1, ells=None, refwmatrix=None, semilogy=True, fn=None):
        k = np.ravel(k)
        if ells is None: ells = sorted(self.kout.keys())

        wmatrix_rotated = self.rotate()[0]
        alphas = np.linspace(1, 0.2, len(k))
        fig, lax = plt.subplots(len(ells), len(ells), sharex=True, sharey=True, figsize=(6, 5))

        for iin, ellin in enumerate(ells):
            for iout, ellout in enumerate(ells):
                ax = lax[iout][iin]
                for ik, kk in enumerate(k):
                    indexout = np.abs(self.kout[ellout] - kk).argmin()
                    # Indices in approximate window matrix
                    norm = self.kin[ellin]
                    if refwmatrix is not None:
                        ax.semilogy(self.kin[ellin], np.abs(refwmatrix[np.ix_(self.mask_ellsout[ellout], self.mask_ellsin[ellin])][indexout, :] / norm), alpha=alphas[ik], color='C0', ls=':', label=r'$W$' if ik == 0 else '')
                    ax.semilogy(self.kin[ellin], np.abs(self.wmatrix.value.T[np.ix_(self.mask_ellsout[ellout], self.mask_ellsin[ellin])][indexout, :] / norm), alpha=alphas[ik], color='C0', label=r'$W^{\mathrm{cut}}$' if ik == 0 else '')
                    ax.semilogy(self.kin[ellin], np.abs(wmatrix_rotated[np.ix_(self.mask_ellsout[ellout], self.mask_ellsin[ellin])][indexout, :] / norm), alpha=alphas[ik], color='C1', label=r'$W^{\mathrm{cut}\prime}$' if ik == 0 else '')                       
                #ax.set_title(r'$\ell_{{\mathrm{{t}}}} = {:d} \times \ell_{{\mathrm{{o}}}} = {:d}$'.format(ellin, ellout))
                text = r'$\ell_{{\mathrm{{t}}}} = {:d} \times \ell_{{\mathrm{{o}}}} = {:d}$'.format(ellin, ellout)
                xlim = ax.get_xlim()
                ax.text((xlim[0]+xlim[1])/2., 1., text, horizontalalignment='center', verticalalignment='top', color='black', fontsize=12)
                ax.set_ylim((1e-4, 2)) 
                if iout == len(ells) - 1: ax.set_xlabel(r'$k_{\mathrm{t}}$ [$h/\mathrm{Mpc}$]')
                if iin == iout == 2: lax[iout][iin].legend(loc='lower right')
        
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.1, hspace=0.12)
        if fn is not None:
            utils.savefig(fn, fig=fig, dpi=300)

    def plot_compactness(self, frac=0.95, ells=None, klim=(0.02, 0.2), refwmatrix=None, fn=None):
        if ells is None:
            ells = sorted(self.kout.keys())
            
        wmatrix_rotated = self.rotate()[0]
        fig, ax = plt.subplots(1, 1, figsize=(4, 3), squeeze=True)

        def compactness(wm, ells, frac):
            weights_bf = sum(np.cumsum(np.abs(wm[np.ix_(self.mask_ellsout[ellout], self.mask_ellsin[ellin])]), axis=-1) for ellin in ells for ellout in ells)
            weights_tot = weights_bf[:, -1]
            iktmax = np.argmax(weights_bf / weights_tot[:, None] >= frac, axis=-1)
            return self.kin[0][iktmax]

        if refwmatrix is not None:
            ax.plot(self.kout[0], compactness(refwmatrix, ells=ells, frac=frac), color='C0', ls=':', label=r'$W$')
        ax.plot(self.kout[0], compactness(self.wmatrix.value.T, ells=ells, frac=frac), color='C0', label=r'$W^{\mathrm{cut}}$')
        ax.plot(self.kout[0], compactness(wmatrix_rotated, ells=ells, frac=frac), color='C1', label=r'$W^{\mathrm{cut}\prime}$')

        for kk in (klim or []): ax.axvline(kk, ls='--', color='k', alpha=0.5)

        ax.set_xlim(xmin=0, xmax=0.4)
        ax.set_ylim(ymin=0)
        ax.plot([0, 0.4], [0, 0.4], ls='-', color='k', alpha=0.5)
        ax.set_xlabel(r'$k_{\mathrm{o}}$ [$h/\mathrm{Mpc}$]')
        ax.set_ylabel(r'$k_{\mathrm{t}}$ [$h/\mathrm{Mpc}$]')
        ax.legend()
        if fn is not None:
            utils.savefig(fn, fig=fig, dpi=300)

    def plot_rotated(self, obs, ells=None, klim=(0.02, 0.2), fn=None):

        if ells is None:
            ells = sorted(self.kout.keys())

        obs_rotated = self.rotate(obs=obs)[2]
        fig, lax = plt.subplots(1, len(ells), figsize=(8, 2.5), sharey=False)

        for ill, ell in enumerate(ells):
            ax = lax[ill]
            ax.plot(self.kout[ell], self.kout[ell] * obs[self.mask_ellsout[ell]], color='C0', label=r'$P(k)$')
            ax.plot(self.kout[ell], self.kout[ell] * obs_rotated[self.mask_ellsout[ell]], color='C1', label=r'$P^{\prime}(k)$')
            ax.set_title(r'$\ell = {}$'.format(ell))
            ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
            ax.set_xlim(klim)

        lax[0].set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        lax[0].legend()
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.2)
        if fn is not None:
            utils.savefig(fn, fig=fig, dpi=300)

    def plot_validation(self, obs, theory, ells=None, klim=(0.02, 0.2), fn=None):

        if ells is None:
            ells = sorted(self.kout.keys())
    
        fig, lax = plt.subplots(2, len(ells), figsize=(8, 3), sharey=False, sharex=True, gridspec_kw={'height_ratios': [4, 2]}, squeeze=False)
        rotate = self.rotate(obs=obs, theory=theory)
        try:
            Wp, Cp, Pp = rotate
            offset = 0.
        except ValueError:
            Wp, Cp, Pp, m, precmatrix, offset = rotate
        std = np.diag(Cp)**0.5
        Ptp = np.matmul(Wp, theory) + offset

        for ill, ell in enumerate(ells):
            color = 'C{}'.format(ill)
            lax[0][ill].errorbar(self.kout[ell], self.kout[ell] * Pp[self.mask_ellsout[ell]], self.kout[ell] * std[self.mask_ellsout[ell]], color=color, marker='.', ls='', label=r'$P^{\prime}(k)$')
            lax[0][ill].plot(self.kout[ell], self.kout[ell] * Ptp[self.mask_ellsout[ell]], color=color, label=r'$W^{\prime}(k, k^{\prime}) P(k^{\prime})$')
            lax[0][ill].set_title(r'$\ell = {}$'.format(ell))

            lax[1][ill].plot(self.kout[ell], (Ptp[self.mask_ellsout[ell]] - Pp[self.mask_ellsout[ell]]) / std[self.mask_ellsout[ell]], color=color)
            lax[1][ill].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
            lax[0][ill].set_xlim(klim)
            lax[1][ill].set_ylim(-2., 2.)

        lax[0][0].set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        lax[1][0].set_ylabel(r'$\Delta P_{\ell}(k) / \sigma$')
        lax[0][0].legend()
        fig.align_ylabels()
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.2, hspace=0.1)
        if fn is not None:
            utils.savefig(fn, fig=fig, dpi=300)

    def __getstate__(self):
        state = {}
        for name in ['kin', 'kout', 'mask_ellsin', 'mask_ellsout', 'khalfout', 'covmatrix', 'mmatrix', 'state', 'csub']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        for name in ['wmatrix']:
            state[name] = getattr(self, name).__getstate__()
        return state
        #return {name: getattr(self, name) for name in ['kin', 'kout', 'mask_ellsin', 'mask_ellsout', 'khalfout', 'wmatrix', 'covmatrix', 'mmatrix', 'state', 'csub'] if hasattr(self, name)}
        
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.wmatrix = BaseMatrix.from_state(self.wmatrix)
      
    
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
    wm = BaseMatrix.load(wm_fn.get_path())
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
    cov = np.loadtxt(cov_fn)
    cov = truncate_cov(cov, kinit=np.arange(0., 0.4, 0.005), kfinal=np.arange(kolim[0], kolim[1], 0.005))
    
    if covtype == 'ezmocks':
        cov_fn = '/global/cfs/cdirs/desi/users/mpinon/Y1/cov/pk/cov_EZmocks_{}_ffa_{}_z{:.3f}-{:.3f}_k{:.2f}-{:.2f}{}.npy'.format(tracer[:7], region, zmin, zmax, kolim[0], kolim[1], '_thetacut{:.2f}'.format(thetacut) if thetacut else '')
        if not os.path.isfile(cov_fn):
            cov = get_EZmocks_covariance(stat='pkpoles', tracer=tracer, region=region, zrange=zrange, completeness='ffa', ells=(0, 2, 4), select=(kolim[0], kolim[1], 0.005), rpcut=rpcut, thetacut=thetacut, return_x=False)
            np.save(cov_fn, cov)
        else:
            print('Loading EZmocks covariance: {}'.format(cov_fn))
            cov = np.load(cov_fn)

    return {'power': pk, 'wmatrix': w, 'covariance': cov}


if __name__ == '__main__':
    
    setup_logging()
    
    source = 'desi' # desi or local
    catalog = 'second' # first, second, or data
    version = 'v4_1'

    tracer = "ELG_LOPnotqso"
    region = "GCcomb"
    zrange = (1.1, 1.6)
    completeness = 'complete'
    
    ells = [0, 2, 4]
    
    kolim = (0., 0.4)
    korebin = 5
    ktmax = 0.5
    ktrebin = 1

    thetacut = 0.05
    
    max_sigma_W = 5
    max_sigma_R = 5
    factor_diff_ell = 10
    csub = False
    covtype = 'analytic'
    
    data = get_data(source=source, catalog=catalog, version=version, tracer=tracer, region=region, zrange=zrange, completeness=completeness, thetacut=thetacut, kolim=kolim, korebin=korebin, ktmax=ktmax, ktrebin=ktrebin, covtype=covtype)
    
    window_rotation = WindowRotation(wmatrix=data['wmatrix'], covmatrix=data['covariance'])
    
    mmatrix, state = window_rotation.fit(Minit='momt' if thetacut else None, max_sigma_W=max_sigma_W, max_sigma_R=max_sigma_R, factor_diff_ell=factor_diff_ell, csub=csub)
    
    output_dir = "/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/{}/rotated_window".format(version)        
    output_fn = LocalFileName().set_default_config(ftype='rotated_all', tracer=tracer, region=region, 
                                                   completeness=completeness, realization=None, weighting=None, 
                                                   thetacut=thetacut)
    
    output_fn.update(fdir=output_dir, zrange=zrange, cellsize=None, boxsize=None, directedges=False)
    output_fn.rotation_attrs['ells'] = ells
    output_fn.rotation_attrs['kobsmax'] = kolim[-1]
    output_fn.rotation_attrs['ktmax'] = ktmax
    output_fn.rotation_attrs['max_sigma_W'] = max_sigma_W
    output_fn.rotation_attrs['max_sigma_R'] = max_sigma_R
    output_fn.rotation_attrs['factor_diff_ell'] = factor_diff_ell
    output_fn.rotation_attrs['covtype'] = covtype
    output_fn.rotation_attrs['csub'] = csub

    window_rotation.save(output_fn.get_path())        
        
