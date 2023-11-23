import os

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

    
def filename(filetype="pk", data_type="Y1secondgenmocks", imock=None, tracer="ELG", completeness="complete_", region="GCcomb", cellsize=None, boxsize=None, rpcut=0, thetacut=0, direct_edges=False, los=None, highres=True, zrange=(0.8, 1.1)):
    if "pk" in filetype:
        fn = '{}_mock{}_{}_{}{}{}{}{}{}{}.npy'.format(filetype, 
                                                        imock, 
                                                        tracer, 
                                                        completeness if (completeness or (data_type=="Y1firstgenmocks" and filetype=='power_spectrum')) else 'ffa_', 
                                                        region, 
                                                        zcut_flag, 
                                                        rpcut_flag, 
                                                        '_directedges_max5000' if direct_edges else '',
                                                        '_los{}'.format(los) if los is not None else '',
                                                         '_highres' if highres else '')