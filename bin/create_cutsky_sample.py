import os
import sys
import argparse
import numpy as np
import fitsio
from mockfactory import Catalog, RandomBoxCatalog, RandomCutskyCatalog, DistanceToRedshift, TabulatedRadialMask, box_to_cutsky, utils, setup_logging
from mockfactory.desi.footprint import is_in_desi_footprint
from cosmoprimo.fiducial import DESI

#from desimodel import io
from desimodel import footprint
#io.load_platescale = load_platescale

os.system('export DESIMODEL=${HOME}/.local/lib/python3.10/site-packages/desimodel')
    
    
def mask_secondgen(nz=0, foot=None, nz_lop=0):
    if foot == 'Y1':
        Y5 = 0
        Y1 = 1
    elif foot == 'Y5':
        Y5 = 1
        Y1 = 0
    else:
        Y5 = 0
        Y1 = 0
    return nz * (2**0) + Y5 * (2**1) + nz_lop * (2**2) + Y1 * (2**3)


def create_randoms(target_type, size, nz_file, zavg=None, zrange=None, boxsize=None, boxcenter=0, seed=0, outfile='randoms_{:d}.fits'.format(0)):
    cosmo = DESI()
    dmin = cosmo.comoving_radial_distance(zrange[0])
    dmax = cosmo.comoving_radial_distance(zrange[1])
    if boxsize is None:
        boxsize = dmax * 3.
        
    #randoms = RandomBoxCatalog(boxsize=boxsize, boxcenter=boxcenter, csize=int(size), seed=seed)
    #randoms_cut = randoms

    #drange, rarange, decrange = box_to_cutsky(boxsize=boxsize, dmax=dmax)
    #rarange = np.array(rarange)
    #rarange[1] -= 1e-9
    #decrange = np.array(decrange)
    #decrange[1] -= 1e-9
    #randoms_cutsky = randoms.cutsky(drange=drange, rarange=rarange, decrange=decrange, noutput=1)
    
    randoms_cutsky = RandomCutskyCatalog(drange=(dmin, dmax), csize=int(size), seed=seed)
    
    # n(z)
    nz = np.loadtxt(nz_file, unpack=True)
    if target_type=="ELG_LOP":
        nz[1] = nz[1] * nz[2]
    mask_radial = TabulatedRadialMask(z=nz[0], nbar=nz[1], interp_order=1, zrange=zrange)
    distance_to_redshift = DistanceToRedshift(distance=cosmo.comoving_radial_distance)
    #randoms_cutsky['Distance'], randoms_cutsky['RA'], randoms_cutsky['DEC'] = utils.cartesian_to_sky(randoms_cutsky.position)
    randoms_cutsky['Z'] = distance_to_redshift(randoms_cutsky['Distance'])
    mask_nz = mask_radial(randoms_cutsky['Z'], seed=seed)
    
    import fitsio
    tiles_fn = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/tiles-DARK.fits'
    tiles = fitsio.read(tiles_fn)
    
    mask_y1 = footprint.is_point_in_desi(tiles, randoms_cutsky['RA'], randoms_cutsky['DEC']) #is_in_desi_footprint(randoms_cutsky['RA'], randoms_cutsky['DEC'], release='y1')
    print('ok')
    mask = mask_nz & mask_y1
    print(randoms_cutsky)
    print(randoms_cutsky[mask])
    #randoms_cutsky[mask].write(outfile)
    

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description='Y1 cutsky mocks')
    parser.add_argument('--todo', type=str, default='data')
    parser.add_argument('--target_type', type=str, default='ELG_LOP', choices=['ELG', 'QSO', 'LRG', 'ELG_LOP', 'ELG_VLO'])
    parser.add_argument('--rmin', type=int, default=0)
    parser.add_argument('--rmax', type=int, default=1)
    parser.add_argument('--realization', type=int, default=0)
    parser.add_argument('--footprint', type=str, default='Y1', choices=['Y1', 'Y5'])
    args = parser.parse_args()
    
    todo = args.todo
    target_type = args.target_type
    rmin = args.rmin
    rmax = args.rmax
    realization = args.realization
    footprint = args.footprint
    
    zrange = {'ELG': (0.8, 1.6), 'LRG': (0.4, 1.1), 'QSO': (0.8, 3.5)}
    zavg = {'ELG': 1.1, 'LRG': 0.8, 'QSO': 1.4}
    nbar = {'ELG': 2.1e-3, 'LRG': 8.4e-4, 'QSO': 1.7e-4}

    parent_path = '/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CutSky/'

    confs = {'ELG': {'downsamp' : 0.7345658717688022, 'input_file' : os.path.join(parent_path,'ELG', 'z1.100', 'cutsky_ELG_z1.100_AbacusSummit_base_c000_ph%s.fits' % str(int(realization)).zfill(3))},
             'LRG': {'downsamp' : 0.708798313382828, 'input_file' : os.path.join(parent_path, 'LRG', 'z0.800', 'cutsky_LRG_z0.800_AbacusSummit_base_c000_ph%s.fits' % str(int(realization)).zfill(3))},
             'QSO': {'downsamp' : 0.39728966594530174, 'input_file' : os.path.join(parent_path, 'QSO', 'z1.400', 'cutsky_QSO_z1.400_AbacusSummit_base_c000_ph%s.fits' % str(int(realization)).zfill(3))}
             }

    outdir = '/global/cfs/cdirs/desi/users/mpinon/secondGenMocksY1/'
    outfile = os.path.join(outdir, 'SeconGen_mock_%s_%d_%s.fits' %(target_type, realization, footprint))   #Where to save it

    down_to_y1 = True  #If the target density should match that of Y1

    
    if todo=='data':
    
        data = fitsio.read(confs[target_type[:3]]['input_file'])

        status = data['STATUS'][()]
        idx = np.arange(len(status))

        mask_main = mask_secondgen(nz=1, foot=footprint)
        idx_main = idx[(status & (mask_main))==mask_main]

        if target_type == 'ELG_LOP' or target_type == 'ELG_VLO':
            mask_LOP = mask_secondgen(nz=1, foot=footprint, nz_lop=1)

            idx_LOP = idx[(status & (mask_LOP))==mask_LOP]
            idx_VLO = np.setdiff1d(idx_main, idx_LOP)

            if down_to_y1:

                ran_lop = np.random.uniform(size = len(idx_LOP))
                idx_LOP = idx_LOP[(ran_lop<=confs[target_type[:3]]['downsamp'])]
                ran_vlo = np.random.uniform(size = len(idx_VLO))
                idx_VLO = idx_VLO[(ran_vlo<=confs[target_type[:3]]['downsamp'])]

            if target_type == 'ELG_LOP':
                idx_main = idx_LOP
            else:
                idx_main = idx_VLO

        else:
            if down_to_y1:
                ran_tot = np.random.uniform(size = len(idx_main))
                idx_main = idx_main[(ran_tot<=confs[target_type[:3]]['downsamp'])]

        data = data[idx_main]

        if os.path.isfile(outfile):
            os.system('rm -f ' + outfile)
        fitsio.write(outfile, data)
        
    if todo=='randoms':
        #nbar = 1e-6
        #boxsize = 10000
        size = 1e8
        
        for i in range(rmin, rmax):
            outfile = os.path.join(outdir, 'randoms_{}_{:d}.fits'.format(target_type, i))
            create_randoms(target_type, size=size, zrange=zrange[target_type[:3]], nz_file=os.path.join(outdir, 'nz/NZ_{}.dat'.format(target_type[:3])), seed=i, outfile=outfile)


if __name__ == "__main__":
    main()