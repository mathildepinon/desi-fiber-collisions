# Adapted from https://github.com/cosmodesi/anotherpipe/tree/main by Pat McDonald

import numpy as np
# if useful, could think of a way to make jax commands work without jax
import jax
import jax.numpy as jnp
from jax import jit
import optax
# not using so don't need to install for now
# import jaxopt
import anotherpipe.utils.jaxopt as jo
# from functools import partial

defaultinitkcut = 0.19
defaultcompind = 21


def Wp_nonorm(M, W): 
    return jnp.matmul(M, W)


def Cp_nonorm(M, C): 
    return jnp.matmul(jnp.matmul(M, C), M.T)


def WpCp(M, W, C, normMask):
    Wpnn = Wp_nonorm(M, W)
    Cpnn = Cp_nonorm(M, C)
    # norm=(Wpnn*normMask).sum(axis=1)
    # return Wpnn/norm[:,None], Cpnn/norm[None,:]/norm[:,None]
    return Wpnn, Cpnn


def softabs(x): 
    return jnp.sqrt(x**2+1e-37)


def lW(W, wW):
    return jnp.sum(softabs(W*wW))/jnp.sum(softabs(W)*(wW > 0))


def lR(C, wR):
    R = RfromC(C)
    return jnp.sum(softabs(R*wR))/jnp.sum(softabs(R)*(wR > 0))


def lM(M, Mtarg):
    # return 0.0*jnp.sum((M-Mtarg)**2)+10*jnp.sum((jnp.sum(M,axis=1)-1)**2)
    return 10*jnp.sum((jnp.sum(M, axis=1)-1)**2)


def lsigsq(C, wsigsq):
    sigsq = jnp.diag(C)
    return jnp.sum(sigsq*wsigsq)


def loss(M, **kwargs):
    wW = kwargs['wW']
    wR = kwargs['wR']
    wsigsq = kwargs['wsigsq']
    normMask = kwargs['normMask']
    Mtarg = kwargs['Mtarg']
    C = kwargs['C']
    if len(M) == 4:
        Mm, mo, mt, m = M
        # N=Mm.shape[1]
        # Mm-=(jnp.sum(Mm,axis=1)[:,None]-1)/N
        W = kwargs['W']
        Wsub = jnp.zeros(W.shape)
        Csub = jnp.zeros(C.shape)
        for tmo, tmt, tm, mask in zip(mo, mt, m, kwargs["momasks"].values()):
            masktmo = mask*tmo
            Wsub += jnp.outer(masktmo, tmt)
            Csub += tm*jnp.outer(masktmo, masktmo)
    else:
        Mm = M[0]
        W = kwargs['W']
        Wsub = 0
        Csub = 0

    Wp, Cp, = WpCp(Mm, W, C, normMask)
    Wp -= Wsub
    Cp -= Csub

    return lW(Wp, wW)+lR(Cp, wR)+lM(Mm, Mtarg)
    # return lW(Wp,wW)+lR(Cp,wR)+lsigsq(Cp,wsigsq)+lM(M,Mtarg)
    # return lW(Wp,wW)+lR(Cp,wR)
    # return lW(Wp,wW)


loss_grad_fn = jax.value_and_grad(loss)


def fit(P, ls=[0], state=None, M=None, compind=defaultcompind, kcut=defaultinitkcut,
        momt=True):
    kwargs = fitkwargs(P, ls=ls)
    if M is None:
        M = initialM(P, ls=ls, compind=compind, kcut=kcut,
                     momt=momt, **kwargs)
    if state is None:
        return jo.metafit(M, lambda M: loss(M, **kwargs))
    return jo.metafit(M, lambda M: loss(M, **kwargs),
                      init_learning_rate=state[0],
                      state=state[2], meta_state=state[1])


def jaxoptfit(P, inM=None, ls=[0], stepsize=0.00002, maxiter=30000):
    if inM is None:
        inM = initialM(**kwargs)
    kwargs = fitkwargs(P, ls=ls)
    print("initial loss:", loss(inM, **kwargs))
    solver = jaxopt.GradientDescent(
        fun=loss, maxiter=maxiter, stepsize=stepsize)
    # solver = jaxopt.GradientDescent(fun=loss,maxiter=20000)
    M, state = solver.run(inM, **kwargs)
    print("final loss:", loss(M, **kwargs))
    return M

# makes no progress
# pg = jaxopt.ProjectedGradient(fun=rw.newloss,
#                              projection=jaxopt.projection.projection_non_negative,
#                             stepsize=0.00002,maxiter=30000)
# posM,state = pg.run(inM,**kwargs)
# rw.newloss(posM,**kwargs)


def optaxfit(M: optax.Params = None, learning_rate=0.00001,
             optimizer: optax.GradientTransformation = None,
             Nsteps=30000, **kwargs) -> optax.Params:
    if optimizer is None:
        # optimizer=optax.yogi(learning_rate=learning_rate)
        # optimizer=optax.adam(learning_rate=learning_rate)
        optimizer = optax.adabelief(learning_rate=learning_rate)
        # optimizer=optax.novograd(learning_rate=learning_rate)
        # optimizer=optax.adamw(learning_rate=learning_rate)

    if M is None:
        M = jnp.identity(kwargs['C'].shape[0])
    opt_state = optimizer.init(M)

    @jax.jit
    def step(M, opt_state, **kwargs):
        loss_value, grads = loss_grad_fn(M, **kwargs)
        updates, opt_state = optimizer.update(grads, opt_state, M)
        M = optax.apply_updates(M, updates)
        return M, opt_state, loss_value
    printstep = max(int(Nsteps/20), 1)
    for i in range(Nsteps):
        M, opt_state, loss_value = step(M, opt_state, **kwargs)
        if i == 0 or i == Nsteps-1 or i == Nsteps-2 or i == Nsteps or i % printstep == 0:
            print(f'step {i}, loss: {loss_value}')

    return M


def setupWeights(P, ls=[0], downfacsigsq=1):
    W = P.W(ls=ls)
    wW = np.zeros_like(W)
    kwid = 0.0036
    ktarg = P.kobs(ls=ls)
    ltarg = P.lobs(ls=ls)
    kt = P.ktheory(ls=ls)
    lt = P.ltheory(ls=ls)
    onekt = P.ppW.xin[0]
    dkt = (onekt[1:]-onekt[:-1])
    dkt = np.append(dkt, [dkt[-1]])
    while dkt.size < kt.size:
        dkt = np.append(dkt, dkt)
    # ktoss=0.35
    # ~infinity
    ktoss = 10.35
    Wpow = 2
    downfacW = 1
    # ~infinity
    capsig = 1000
    # could make next even larger...
    difflfac = 100
    for i, kd in enumerate(ktarg):
        wW[i, :] = downfacW*np.minimum(((kt-kd)/kwid)**Wpow, capsig**Wpow)
        # wW[i,:]+=difflfac*(ltarg[i]-lt)**2
        wW[i, :] += difflfac*(ltarg[i] != lt)
        if kd > ktoss:
            wW[i, kt > ktoss] = 0

    Rpow = 2
    C = P.C(ls=ls)
    wR = np.zeros_like(C)
    sig = np.sqrt(np.diag(C))
    wsigsq = np.zeros_like(sig)
    downfacR = 1
    for i, kd in enumerate(ktarg):
        wR[i, :] = downfacR*np.minimum(((ktarg-kd)/kwid)**Rpow, capsig**Rpow)
        # wR[i,:]+=difflfac*(ltarg[i]-ltarg)**2
        wR[i, :] += difflfac*(ltarg[i] != ltarg)
        wsigsq[i] = downfacsigsq/sig[i]**2
        if kd > ktoss:
            wR[i, ktarg > ktoss] = 0
            wsigsq[i] = 0

    wsigsq /= np.sum(wsigsq > 0)
    return jax.device_put(wW), jax.device_put(wR), jax.device_put(wsigsq)


def initialM(P, ls=[0], momt=True, maxoff=0, compind=defaultcompind,
             kcut=defaultinitkcut, **kwargs):
    C = kwargs["C"]
    # might imagine this was a good idea, but doesn't seem to be...
    # Meig=np.linalg.eig(C).eigenvectors.T
    N = C.shape[0]
    Mm = jnp.identity(N, dtype=jnp.float32)
    offs = jnp.zeros((N, N), dtype=jnp.float32)
    for k in range(1, maxoff+1):
        offs += jnp.eye(N, k=k, dtype=jnp.float32) + \
            jnp.eye(N, k=-k, dtype=jnp.float32)
    # for k in range(1,maxoff+1):
    #  offs+=jnp.eye(N,k=k,dtype=jnp.float32)
    ltarg_s = P.lobs_series(ls=ls)
    ltarg = ltarg_s.values
    samelmask = jnp.tile(ltarg, (N, 1)) == jnp.tile(ltarg[:, None], (1, N))
    offs *= samelmask
    sumoff = jnp.sum(offs, axis=1)
    offs /= jnp.maximum(sumoff, 1)[:, None]
    Mm -= offs

    mo = ()
    mt = ()
    m = ()
    W = P.W(ls=ls)
    for l, mask in kwargs["momasks"].items():
        compind = P.lobs_series(ls=[l]).index[compind]
        comprow = W.loc[compind]
        mt += (jnp.array((comprow*(P.ktheory(ls=ls) > kcut)).values),)
        mo += (jnp.array([row[ltarg_s[i]].iloc[-1]/comprow[ltarg_s[i]].iloc[-1]
               for i, row in W.iterrows()]*mask),)
        m += (0.0,)
    if momt:
        return [Mm, mo, mt, m]
    return [Mm]


def fitkwargs(P, ls=[0], sigsqweight=1):
    W = P.W(ls=ls).to_numpy()
    normMask = jax.device_put(P.WnormMask(ls=ls))
    C = P.C(ls=ls)
    wW, wR, wsigsq = setupWeights(P, ls=ls, downfacsigsq=sigsqweight)
    Mtarg = jnp.identity(C.shape[0])
    momasks = {l: P.lomask(l, ls=ls) for l in ls}
    return {'wW': wW, 'W': W, 'wR': wR, 'C': C, 'wsigsq': wsigsq, 'normMask': normMask,
            'Mtarg': Mtarg, 'momasks': momasks}


def orgloss(M, **kwargs):
    targW = kwargs['targW']
    W = kwargs['W']
    targR = kwargs['targR']
    C = kwargs['C']
    return jnp.sum((targW-Wp(M, W))**2)+jnp.sum((targR-Rp(M, C))**2)


def RfromC(C):
    sig = jnp.sqrt(jnp.diag(C))
    denom = jnp.outer(sig, sig)
    return C/denom


def Rp(M, C): return RfromC(Cp(M, C))


def gdupdate_params(M, learning_rate, grads):
    M = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g, M, grads)
    return M

# @partial(jax.jit, static_argnums=(1,))


@jit
def gdstep(M, learning_rate=0.09, **kwargs):
    loss_val, grads = loss_grad_fn(M, **kwargs)
    M = update_params(M, learning_rate, grads)
    return M, loss_val


def gdfit(learning_rate=0.09, Nstep=101, M=None, **kwargs):
    if M is None:
        M = jnp.identity(kwargs['C'].shape[0], dtype=jnp.float32)
    printstep = max(int(Nstep/20), 1)
    for i in range(Nstep):
        M, loss_val = step(M, learning_rate, **kwargs)
        if i == 0 or i == Nstep-1 or i % printstep == 0:
            print(f'Loss step {i}: ', loss_val)
    return M
