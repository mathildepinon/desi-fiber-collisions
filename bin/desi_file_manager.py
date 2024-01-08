import os
import copy
import numpy as np
    

class DESIFileName():
    """
    Class to manage filenames in DESI Y1 clustering measurementss.
 
    Attributes
    ----------
    fdir : str, default=''
        Directory.
        
    mocktype : str, default='SecondGenMocks/AbacusSummit'
        Mock type, e.g. 'SecondGenMocks/AbacusSummit' or 'SecondGenMocks/EZmock'.
        
    version : str, default='v3'
        Mocks version.
        
    ftype : str, default='pkpoles'
        File type, e.g. 'pkpoles', 'wmatrix_smooth' etc.
        
    realization : int, default=0
        Index of the realization (e.g. mock index). If None, merged realizations.
        
    tracer : str, default='ELG'
        One of ELG, LRG, QSO, BGS.
    
    completeness : bool or str, default=True
        If boolean, whether file is complete or fiber assigned catalog. If string, shortname for the fiber assignment applied to file (e.g. 'complete', 'ffa', 'altmtl').
        
    region : str, default='GCcomb'
        Galactic cap (one of SGC, NGC, GCcomb).
        
    **kwargs : other keyword arguments.
    """
    _defaults = dict(fdir='', mocktype='SecondGenMocks/AbacusSummit', version="v3", ftype="pkpoles", realization=0, tracer="ELG", completeness=True, region="GCcomb", zrange=None, weighting=None, nran=None, cellsize=None, nmesh=None, boxsize=None, njack=None, split=None, rpcut=0., thetacut=0.)

    def __init__(self, *args, **kwargs):
        if len(args):
            if isinstance(args[0], self.__class__):
                self.__dict__.update(args[0].__dict__)
                return
            try:
                kwargs = {**args[0], **kwargs}
            except TypeError:
                args = dict(zip(self._defaults, args))
                kwargs = {**args, **kwargs}
        for name, value in self._defaults.items():
            setattr(self, name, copy.copy(value))
        self.update(**kwargs)

    def update(self, **kwargs):
        """Update input attributes."""
        for name, value in kwargs.items():
            if name in self._defaults:
                setattr(self, name, copy.copy(value))
            else:
                raise ValueError('Unknown argument {}; supports {}'.format(name, list(self._defaults)))
                
    def get_path(self, **kwargs):
        """Get path corresponding to input attributes."""
        self.update(**kwargs)
        
        # NB: either rp- or theta-cut (not both!)
        if bool(self.rpcut):
            cut = '_rpcut{:.1f}'.format(self.rpcut)
        elif bool(self.thetacut):
            cut = '_thetacut{:.2f}'.format(self.thetacut)
        else:
            cut = ''
        
        # completeness
        if isinstance(self.completeness, str):
            comp = self.completeness
        else:
            comp = 'complete' if self.completeness else 'altmtl'
            
        # realization
        if isinstance(self.realization, str):
            rlz = self.realization
        elif self.realization is None:
            rlz = 'merged'
        else:
            rlz = 'mock{:d}'.format(self.realization)
        
        # for additional options
        def opt_attr(name, attr):
            if (attr is None):
                return ''
            else:
                return '_{}{}'.format(name, attr)
            
        if ('xi' in self.ftype) or ('counts' in self.ftype):
            self.fdir = os.path.join(self.fdir, '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/{}/desipipe/{}/{}/baseline_2pt/{}/xi/smu'.format(self.mocktype, self.version, comp, rlz))
        else:
            self.fdir = os.path.join(self.fdir, '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/{}/desipipe/{}/{}/baseline_2pt/{}/pk/'.format(self.mocktype, self.version, comp, rlz))

        fname = '{}_{}_{}_z{:.1f}-{:.1f}{}{}{}{}{}{}{}.npy'.format(self.ftype, self.tracer, self.region, self.zrange[0], self.zrange[1], self.weighting if self.weighting is not None else '', opt_attr('nran', self.nran), opt_attr('njack', self.njack), opt_attr('split', self.split), opt_attr('cellsize', self.cellsize), opt_attr('boxsize', self.boxsize), cut)

        return os.path.join(self.fdir, fname)
                
    
    def set_default_config(self, **kwargs):
        self.update(**kwargs)
            
        if self.tracer[:3]=='ELG':
            if 'allcounts' in self.ftype:
                default_options = dict(tracer='ELG_LOP' if (self.completeness or 'EZmock' in self.mocktype) else 'ELG_LOPnotqso',
                                       zrange=(0.8, 1.6) if self.zrange is None else self.zrange)
                                       #nran=10,
                                       #njack=0,
                                       #split=20)
            else:
                default_options = dict(tracer='ELG_LOP' if (self.completeness or 'EZmock' in self.mocktype) else 'ELG_LOPnotqso',
                                       zrange=(0.8, 1.6) if self.zrange is None else self.zrange)
                                       #nran=18,
                                       #cellsize=6,
                                       #boxsize=9000)
                            
        elif self.tracer[:3]=='LRG':
            default_options = dict(tracer='LRG',
                                   zrange=(0.4, 0.6) if self.zrange is None else self.zrange)      
                
        ## need to add QSO, BGS
        else:
            raise ValueError('Unknown tracer: {}'.format(tracer))
        
        self.update(**default_options)
        
        return self