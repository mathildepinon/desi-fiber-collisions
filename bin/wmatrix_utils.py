import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from pypower import utils

def plot_matrix(mat, x1=None, x2=None, xlabel1=None, xlabel2=None, barlabel=None, label1=None, label2=None,
                corrcoef=False, figsize=None, norm=None, labelsize=None):

    size1, size2 = [row[0].shape[0] for row in mat], [col.shape[1] for col in mat[0]]
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def _make_list(x, size):
        if not utils.is_sequence(x):
            x = [x] * size
        return list(x)

    if x2 is None: x2 = x1
    x1, x2 = [_make_list(x, len(size)) for x, size in zip([x1, x2], [size1, size2])]
    if xlabel2 is None: xlabel2 = xlabel1
    xlabel1, xlabel2 = [_make_list(x, len(size)) for x, size in zip([xlabel1, xlabel2], [size1, size2])]
    if label2 is None: label2 = label1
    label1, label2 = [_make_list(x, len(size)) for x, size in zip([label1, label2], [size1, size2])]

    if corrcoef:
        mat = utils.cov_to_corrcoef(np.bmat(mat).A)
        cumsize1, cumsize2 = [np.insert(np.cumsum(size), 0, 0) for size in [size1, size2]]
        mat = [[mat[start1:stop1, start2:stop2] for start2, stop2 in zip(cumsize2[:-1], cumsize2[1:])] for start1, stop1 in zip(cumsize1[:-1], cumsize1[1:])]

    norm = norm or Normalize(vmin=min(item.min() for row in mat for item in row), vmax=max(item.max() for row in mat for item in row))
    nrows, ncols = [len(x) for x in [size2, size1]]
    figsize = figsize or tuple(max(n * 3, 6) for n in [ncols, nrows])
    if np.ndim(figsize) == 0: figsize = (figsize,) * 2
    xextend = 0.8
    fig, lax = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False,
                            figsize=(figsize[0] / xextend, figsize[1]),
                            gridspec_kw={'width_ratios': size1, 'height_ratios': size2[::-1]},
                            squeeze=False)
    for i in range(ncols):
        for j in range(nrows):
            ax = lax[nrows - j - 1][i]
            xx1, xx2 = x1[i], x2[j]
            if x1[i] is None: xx1 = 1 + np.arange(mat[i][j].shape[0])
            if x2[j] is None: xx2 = 1 + np.arange(mat[i][j].shape[1])
            mesh = ax.pcolor(xx1, xx2, mat[i][j].T, norm=norm, cmap=plt.get_cmap('RdBu'))
            if i > 0 or x1[i] is None: ax.yaxis.set_visible(False)
            if j == 0 and xlabel1[i]: ax.set_xlabel(xlabel1[i])
            if j > 0 or x2[j] is None: ax.xaxis.set_visible(False)
            if i == 0 and xlabel2[j]: ax.set_ylabel(xlabel2[j])
            ax.tick_params()
            if label1[i] is not None or label2[j] is not None:
                text = r'{} $\times$ {}'.format(label1[i], label2[j])
                ax.text(0.05, 0.95, text, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, color='black', fontsize=12)
            ax.grid(False)
            
    plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0.15, bottom=0.15, right=0.8)
    cbar_ax = fig.add_axes([xextend+0.025, 0.15, 0.02, 0.8])
    cbar = fig.colorbar(mesh, cax=cbar_ax)
    if barlabel: cbar.set_label(barlabel)
    fig.align_ylabels()
    return lax

# compute window matrix from window power
def compute_wmatrix(windows, power, output=None, ellsin=(0, 2, 4)):
    import numpy as np
    from pypower import PowerSpectrumSmoothWindow, PowerSpectrumOddWideAngleMatrix, PowerSpectrumSmoothWindowMatrix, PowerSpectrumStatistics
    from desi_y1_files import load

    windows = load(windows, load=PowerSpectrumSmoothWindow.load)
    if isinstance(windows, (tuple, list)):
        argsort = np.argsort([np.max(window.attrs['boxsize']) for window in windows])[::-1]
        windows = [windows[ii] for ii in argsort]
        window = windows[0].concatenate_x(*windows, frac_nyq=0.9)
    else:
        window = windows

    power = load(power, load=PowerSpectrumStatistics.load)

    # Let us compute the wide-angle and window function matrix
    ellsin = list(ellsin) # input (theory) multipoles
    wa_orders = 1 # wide-angle order
    sep = np.geomspace(1e-4, 1e5, 1024 * 16) # configuration space separation for FFTlog
    kin_rebin = 2 # rebin input theory to save memory
    #sep = np.geomspace(1e-4, 2e4, 1024 * 16) # configuration space separation for FFTlog, 2e4 > sqrt(3) * 8000
    #kin_rebin = 4 # rebin input theory to save memory
    kin_lim = (0, 2e1) # pre-cut input (theory) ks to save some memory
    # Input projections for window function matrix:
    # theory multipoles at wa_order = 0, and wide-angle terms at wa_order = 1
    projsin = ellsin + PowerSpectrumOddWideAngleMatrix.propose_out(ellsin, wa_orders=wa_orders)
    # Window matrix
    wmatrix = PowerSpectrumSmoothWindowMatrix(power.k, projsin=projsin, projsout=power.ells, window=window, sep=sep, kin_rebin=kin_rebin, kin_lim=kin_lim)
    # We resum over theory odd-wide angle
    wmatrix.resum_input_odd_wide_angle()
    wmatrix.attrs.update(power.attrs)
    if output is not None: output.save(wmatrix)
    return wmatrix