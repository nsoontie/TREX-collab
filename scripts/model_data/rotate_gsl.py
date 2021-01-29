# Script to rotate gsl500m currents to east/north
# Nancy Soontiens

import glob
import os

import numpy as np
import xarray as xr


SRC_DIR='/home/soontiensn/data/MEOPAR-TREX/gsl500/model/three-hourly'
SAV_DIR='/home/soontiensn/data/MEOPAR-TREX/gsl500/model/three-hourly-rotated'
rotation_angle='/home/soontiensn/data/MEOPAR-TREX/gsl500/grid/angle_file.nc'

dang = xr.open_dataset(rotation_angle)
theta = np.squeeze(dang.LAAN.values)
costheta = np.cos(theta)
sintheta = np.sin(theta)

files = glob.glob(os.path.join(SRC_DIR, '*.nc'))
files.sort()

drop = ['tauvo', 'tauuo', 'zosvar', 'tosvar', 'sbt', 'sowindsp', 'taum',
        'sosflxdo', 'tohfls', 'wfo', 'mldr10_1', 'mldkz5', 'runoffs', 'qlw_oce',
        'qsb_oce', 'qla_oce', 'qsr_oce', 'qns_oce', 'qsr', 'qns', 'precip', 'SSHT',
        'SSHP', 'zos']

for f in files:
    basename = os.path.basename(f)
    newfile = os.path.join(SAV_DIR, 'rotated_{}'.format(basename))
    print(basename)

    d = xr.open_dataset(f)
    ueast = d.uo.values*costheta - d.vo.values*sintheta
    vnorth = d.uo.values*sintheta + d.vo.values*costheta
    d.uo.values = ueast
    d.uo.attrs['long_name'] = 'zonal surface current'
    d.vo.values = vnorth
    d.vo.attrs['long_name'] = 'meridional surface current'

    u15_east = d._U15.values*costheta - d._V15.values*sintheta
    v15_north = d._U15.values*sintheta + d._V15.values*costheta
    d._U15.values = u15_east
    d._U15.attrs['long_name'] = 'zonal 15m current'
    d._U15.attrs['units'] = 'm/s'

    d._V15.values = v15_north
    d._V15.attrs['long_name'] ='meridional 15m current'
    d._V15.attrs['units'] = 'm/s'

    d = d.drop(drop)
    print('saving {}'.format(newfile))
    d.to_netcdf(newfile)
            
