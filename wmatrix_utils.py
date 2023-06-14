import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from pypower import utils

def plot_matrix(mat, x1=None, x2=None, xlabel1=None, xlabel2=None, barlabel=None, label1=None, label2=None,
                corrcoef=False, figsize=None, norm=None, labelsize=None):

    size1, size2 = [row[0].shape[0] for row in mat], [col.shape[1] for col in mat[0]]

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
    wspace = hspace = 0.18
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    for i in range(ncols):
        for j in range(nrows):
            ax = lax[nrows - j - 1][i]
            xx1, xx2 = x1[i], x2[j]
            if x1[i] is None: xx1 = 1 + np.arange(mat[i][j].shape[0])
            if x2[j] is None: xx2 = 1 + np.arange(mat[i][j].shape[1])
            mesh = ax.pcolor(xx1, xx2, mat[i][j].T, norm=norm, cmap=plt.get_cmap('jet_r'))
            if i > 0 or x1[i] is None: ax.yaxis.set_visible(False)
            if j == 0 and xlabel1[i]: ax.set_xlabel(xlabel1[i], fontsize=labelsize)
            if j > 0 or x2[j] is None: ax.xaxis.set_visible(False)
            if i == 0 and xlabel2[j]: ax.set_ylabel(xlabel2[j], fontsize=labelsize)
            ax.tick_params()
            if label1[i] is not None or label2[j] is not None:
                text = r'{} $\times$ {}'.format(label1[i], label2[j])
                ax.text(0.05, 0.95, text, horizontalalignment='left', verticalalignment='top',\
                        transform=ax.transAxes, color='black')

    fig.subplots_adjust(right=xextend)
    cbar_ax = fig.add_axes([xextend + 0.25, 0.15, 0.03, 0.7])
    cbar_ax.tick_params()
    cbar = fig.colorbar(mesh, cax=cbar_ax)
    if barlabel: cbar.set_label(barlabel, rotation=90)
    return lax