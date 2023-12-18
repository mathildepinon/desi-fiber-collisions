import os
import copy
import numpy as np

def naming(filetype="power", data_type="Y1secondgenmocks", imock=None, tracer="ELG", completeness="complete_", region="GCcomb", cellsize=None, boxsize=None, rpcut=0, thetacut=0, direct_edges=False, los=None, highres=True, zrange=(0.8, 1.1)):
    if "power" in filetype:
        zcut_flag = '_zcut' if (completeness and data_type=="Y1firstgenmocks") else ''
        if (data_type=="Y1firstgenmocks") and rpcut:
            rpcut_flag = '_th{:.1f}'.format(rpcut)
        else:
            if rpcut:
                rpcut_flag = '_rpcut{:.1f}'.format(rpcut)
            elif thetacut:
                rpcut_flag = '_thetacut{:.2f}'.format(thetacut)
            else:
                rpcut_flag = ''
        if data_type == "rawY1secondgenmocks":
            return '{}_rawcutsky_mock{}_{}_{}_cellsize{:d}{}{}.npy'.format(filetype, 
                                                    imock, 
                                                    tracer,
                                                    region,
                                                    cellsize,
                                                    rpcut_flag,
                                                    '_directedges_max5000' if direct_edges else ''#,
                                                    #'_highres' if highres else '',
                                                    )
        elif data_type == "cubicsecondgenmocks":
            return '{}_mock{}_{}{}{}{}{}.npy'.format(filetype, 
                                                    imock, 
                                                    tracer, 
                                                    rpcut_flag,
                                                    '_directedges_max5000' if direct_edges else '',
                                                    '_los{}'.format(los) if los is not None else '',
                                                    '_highres' if highres else '')
        elif data_type == "Y1firstgenmocks" or data_type == "Y1secondgenmocks":
            return '{}_mock{}_{}_{}{}{}{}{}{}{}.npy'.format(filetype, 
                                                        imock, 
                                                        tracer, 
                                                        completeness if (completeness or (data_type=="Y1firstgenmocks" and filetype=='power_spectrum')) else 'ffa_', 
                                                        region, 
                                                        zcut_flag, 
                                                        rpcut_flag, 
                                                        '_directedges_max5000' if direct_edges else '',
                                                        '_los{}'.format(los) if los is not None else '',
                                                         '_highres' if highres else '')
        elif data_type == "Y1secondgenmocksOFF":
            mock_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks/AbacusSummit/'
            return os.path.join(mock_dir, 'mock{}/pk/pkpoles_{}_{}gtlimaging_{}_{:.1f}_{:.1f}_default_lin{}.npy'.format(imock, tracer, completeness, region, zrange[0], zrange[1], '_rpcut{:.1f}'.format(rpcut) if rpcut else ''))
    
    if filetype in ["window", "wmatrix", "wm"]:
        if data_type=="Y1firstgenmocks":
            rpcut_flag = '_rp{:.1f}'.format(rpcut) if rpcut else ''
        else:
            if rpcut:
                rpcut_flag = '_rpcut{:.1f}'.format(rpcut)
            elif thetacut:
                rpcut_flag = '_thetacut{:.2f}'.format(thetacut)
            else:
                rpcut_flag = ''
        if data_type == "rawY1secondgenmocks":
            completeness = ''
        link='_'
        if data_type == 'Y1secondgenmocksOFF':
            windows_dir = '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/windows'
            return os.path.join(windows_dir, 'wmatrix_smooth_{}_{}gtlimaging_{}_{:.1f}_{:.1f}_default_lin{}{}.npy'.format(tracer, completeness, region, zrange[0], zrange[1], '_rpcut{:.1f}'.format(rpcut) if rpcut else '', '_directedges' if direct_edges else ""))
        return '{}{}{}{}{}_{}{}{}{}{}.npy'.format(filetype, 
                                               '{}' if boxsize is None else '_boxsize{:d}'.format(int(boxsize)),
                                               '_cellsize{:d}'.format(cellsize) if cellsize is not None else '',
                                               '_rawcutsky' if data_type == "rawY1secondgenmocks" else '',
                                               '_mock{:d}'.format(imock) if (imock is not None) else '',
                                               tracer+link,
                                               completeness,
                                               region,
                                               rpcut_flag,
                                               '_directedges_max5000' if direct_edges else '')

    

class LocalFileName():
    """
    Class to manage my filenames for DESI Y1 clustering/fiber collisions.
 
    Attributes
    ----------
    fdir : str, default=''
        Derectory where file is located.
        
    mockgen : str, default='second'
        Generation of Abacus simulation, either 'first', 'second', or 'cubic'.
        
    ftype : str, default='power'
        File type, e.g. 'pkpoles', 'power', 'window' etc. Note: 'pkpoles' files are instances of pypower.PowerSpectrumStatistics whereas 'power' files are pypower.CatalogFFTPower. If ftype includes 'sculpt' (e.g. 'wmatrix_smooth_sculpt'), take corresponding file with Pat's window transformation.
        
    realization : int, default=None
        Index of the realization (e.g. mock index). If None, merged realizations.
        
    tracer : str, default='ELG'
        One of 'ELG', 'LRG, 'QSO', 'BGS'.
    
    completeness : bool or str, default=True
        If boolean, whether file is complete or fiber assigned catalog. If string, shortname for the fiber assignment applied to file (e.g. 'complete', 'ffa').
        
    region : str, default='GCcomb'
        Galactic cap (one of SGC, NGC, GCcomb).
        
    zrange : tuple, default=None
        Redshift range.
        
    **kargs: other keyword arguments.
    """
    _defaults = dict(fdir='', mockgen='second', ftype="power", realization=0, tracer="ELG", completeness=True, region="GCcomb", zrange=None, z=None, weighting=None, nran=None, los=None, cellsize=None, nmesh=None, boxsize=None, rpcut=0., thetacut=0., directedges=True, sculpt_attrs=None)

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
        
        def opt_attr(name, attr):
            if (attr is None):
                return ''
            else:
                return '_{}{}'.format(name, attr)
            
        #NB: either rp- or theta-cut (not both!)
        if bool(self.rpcut):
            cut = '_rpcut{:.1f}'.format(self.rpcut)
        elif bool(self.thetacut):
            cut = '_thetacut{:.2f}'.format(self.thetacut)
        else:
            cut = ''
                            
        if self.fdir=='':
            self.fdir = os.path.join('/global/cfs/cdirs/desi/users/mpinon/{}GenMocksY1'.format(self.mockgen))
            if 'corr' in self.ftype:
                self.fdir = os.path.join(self.fdir, 'xi')
            else:
                self.fdir = os.path.join(self.fdir, 'pk')

        if isinstance(self.completeness, str):
            comp = self.completeness
        else:
            if 'first' in self.mockgen:
                comp = '_complete' if self.completeness else ''
            elif 'second' in self.mockgen:
                comp = '_complete' if self.completeness else '_ffa'
            elif 'cubic' in self.mockgen:
                comp = ''
        if self.zrange is not None:
            z = '_z{:.1f}-{:.1f}'.format(self.zrange[0], self.zrange[1])
        else:
            if self.z is not None:
                z = '_z{:.3f}'.format(self.z)
            else:
                z = ''
        if bool(cut) & self.directedges:
            direct = '_directedges_max5000'
        else:
            direct = ''
            
        # when sculpting window (chnage of basis through Pat's transformation)
        if self.sculpt_attrs is not None:
            sculpt_attrs = '_ells{}_kobsmax{:.1f}_ktmax{:.1f}_capsig{:d}_difflfac{:d}_{}cov'.format(''.join([str(i) for i in self.sculpt_attrs['ells']]), self.sculpt_attrs['kobsmax'], self.sculpt_attrs['ktmax'], self.sculpt_attrs['capsig'], self.sculpt_attrs['difflfac'], self.sculpt_attrs['covtype'])
        else:
            sculpt_attrs = ''

        fname = '{}{}_{}{}{}{}{}{}{}{}{}{}{}{}{}.npy'.format(self.ftype, opt_attr('mock', self.realization), self.tracer, comp, opt_attr('', self.region), z, self.weighting if self.weighting is not None else '', opt_attr('nran', self.nran), opt_attr('los', self.los), opt_attr('cellsize', self.cellsize), opt_attr('nmesh', self.nmesh), opt_attr('boxsize', self.boxsize), cut, direct, sculpt_attrs)
            
        return os.path.join(self.fdir, fname)
        
    def set_default_config(self, **kwargs):
        self.update(**kwargs)
            
        if self.mockgen == 'first':
            if self.tracer[:3]=='ELG':
                if 'corr' in self.ftype:
                    default_options = dict(tracer='ELG',
                                           zrange=(0.8, 1.6))
                    subdir = 'xi'
                else:
                    default_options = dict(tracer='ELG',
                                           zrange=(0.8, 1.6),
                                           directedges=False) 
                    subdir = 'pk'
            ## need to add LRG, QSO, BGS
            else:
                raise ValueError('Unknown tracer: {}'.format(tracer))
                
            default_options['fdir'] = os.path.join('/global/cfs/cdirs/desi/users/mpinon/firstGenMocksY1', subdir)

        if self.mockgen == 'second':
            if self.tracer[:3]=='ELG':
                if 'corr' in self.ftype:
                    default_options = dict(tracer='ELG_LOP',
                                           zrange=(0.8, 1.6))
                    subdir = 'xi'
                else:
                    default_options = dict(tracer='ELG_LOP',
                                           zrange=(0.8, 1.6),
                                           cellsize=4,
                                           directedges=True)
                    subdir = 'pk'
                    
                    if 'sculpt' in self.ftype:
                        default_options['sculpt_attrs'] = {'ells': [0, 2, 4], 'kobsmax': 0.4, 'ktmax': 0.5, 'capsig': 5, 'difflfac': 10, 'covtype': 'analytic'}
                        
            ## need to add LRG, QSO, BGS
            else:
                raise ValueError('Unknown tracer: {}'.format(tracer))
 
            default_options['fdir'] = os.path.join('/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1', subdir)
        
        if self.mockgen == 'cubic':
            if self.tracer[:3]=='ELG':
                default_options = dict(tracer='ELG',
                                       region=None,
                                       zrange=None,
                                       z=1.1,
                                       nmesh=2048,
                                       boxsize=2000)
                subdir = 'pk'
            ## need to add LRG, QSO, BGS
            else:
                raise ValueError('Unknown tracer: {}'.format(tracer))
            
            default_options['fdir'] = os.path.join('/global/cfs/cdirs/desi/users/mpinon/cubicSecondGenMocks', subdir)

        self.update(**default_options)
        
        return self

