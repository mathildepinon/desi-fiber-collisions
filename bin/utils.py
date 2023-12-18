import os
import copy
import numpy as np

def load_poles(fn):
    from pypower import MeshFFTPower, PowerSpectrumMultipoles  
    toret = MeshFFTPower.load(fn)
    if hasattr(toret, 'poles'):
        toret = toret.poles
    else:
        toret = PowerSpectrumMultipoles.load(fn)
    return toret

def load_poles_list(fns, xlim=None, rebin=None):
    list_data, list_shotnoise = [], []
    x, xedges, ells = None, None, None
    
    for mock in fns:
        if os.path.exists(mock):
            poles = load_poles(mock)
        else:
            raise FileNotFoundError('No such file or directory: {}'.format(mock))
        mock_shotnoise = poles.shotnoise
        
        if xlim is None:
            xlim = {ell: (0, np.inf) for ell in poles.ells}
        mock_ells, mock_x, mock_edges, mock_data = [], [], [], []
        for ell, lim in xlim.items():
            poles_slice = poles.copy().select(lim)
            if rebin is not None:
                poles_slice.slice(slice(0, len(poles_slice.k) // rebin * rebin, rebin))
            mock_ells.append(ell)
            mock_edges.append(poles_slice.edges[0])
            mock_x.append(poles_slice.modeavg())
        mock_data.append(poles_slice(ell=poles.ells, complex=False))
        x, edges, ells = mock_x, mock_edges, mock_ells
        if not all(np.allclose(sx, mx, atol=0., rtol=1e-3) for sx, mx in zip(edges, mock_edges)):
            raise ValueError('{} does not have expected k-edges (based on previous data)'.format(mock))
        if mock_ells != ells:
            raise ValueError('{} does not have expected poles (based on previous data)'.format(mock))

        list_data.append(np.concatenate(mock_data))
        list_shotnoise.append(mock_shotnoise)

    data = np.mean(list_data, axis=0)
    shotnoise = np.mean(list_shotnoise, axis=0)

    toret = dict(data=data,
                 shotnoise=shotnoise,
                 ells=ells,
                 k=x,
                 edges=edges)
    
    return toret

def load_corr_list(fns, ells=(0, 2, 4), xlim=None, rebin=None):
    list_data = []
    x, xedges = None, None
    
    from pycorr import TwoPointCorrelationFunction
    
    for mock in fns:
        if os.path.exists(mock):
            poles = TwoPointCorrelationFunction.load(mock)
        else:
            raise FileNotFoundError('No such file or directory: {}'.format(mock))
        
        if xlim is None:
            xlim = {ell: (0, np.inf) for ell in ells}
        mock_ells, mock_x, mock_edges, mock_data = [], [], [], []
        for ell, lim in xlim.items():
            poles_slice = poles.copy().select(lim)
            if rebin is not None:
                poles_slice.slice(slice(0, len(poles_slice.sep) // rebin * rebin, rebin))
            mock_ells.append(ell)
            mock_x.append(poles_slice.sepavg())
            mock_edges.append(poles_slice.edges[0])
        mock_data.append(poles_slice.get_corr(ell=ells, return_sep=False, ignore_nan=True))
        x, edges, ells = mock_x, mock_edges, mock_ells
        if not all(np.allclose(sx, mx, atol=0., rtol=1e-3) for sx, mx in zip(edges, mock_edges)):
            raise ValueError('{} does not have expected s-edges (based on previous data)'.format(mock))
        if mock_ells != ells:
            raise ValueError('{} does not have expected poles (based on previous data)'.format(mock))

        list_data.append(np.concatenate(mock_data))

    data = np.mean(list_data, axis=0)

    toret = dict(data=data,
                 ells=ells,
                 sep=x,
                 edges=edges)
    
    return toret
