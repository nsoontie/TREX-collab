# Script to interpolate CIOPS-E data to a regular latlon grid

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.interpolate as interp
import xarray as xr

from driftutils.rotate_fields import rot_rep_2017_p
SRC_DIR='/home/nso001/data/work2/models/ciops-e_operational_forecasts/pseudo-analysis'
OUT_DIR='/home/nso001/data/work2/models/ciops-e_operational_forecasts/pseudo-analysis_interp/output'
MESH='/home/sdfo000/sitestore4/opp_drift_fa3/share_drift/CIOPSE_SN1500/CIOPSE_mesh_files/mesh_mask_NWA36_Bathymetry_flatbdy_20181109_3_filter_min_7p5.nc'
rotation_file='/home/nso001/data/work2/rotation_pickles/ciopse/ciopse.rotation.pickle'

UVEL = 'uos'
VVEL = 'vos'
ULAT = 'gphiu'
ULON = 'glamu'
VLAT = 'gphiv'
VLON = 'glamv'

DATE='202010*'

# Set up new grid
LON_MIN, LON_MAX = -71, -56
LAT_MIN, LAT_MAX = 45, 51
DELTA=0.03 # lat/lon grid spacing for new grid

def main():
    # First, set up new grid and masks
    loni = np.arange(LON_MIN, LON_MAX, DELTA)
    lati = np.arange(LAT_MIN, LAT_MAX, DELTA)
    loni, lati = np.meshgrid(loni, lati)

    mesh = xr.open_dataset(MESH) 
    umask = mesh.umask.values[0,0,...]
    vmask = mesh.vmask.values[0,0,...]

    # Interpolate masks to new grid
    pointsU = (mesh[ULON].values.flatten(), mesh[ULAT].values.flatten())
    umaski = interp.griddata(pointsU, 1 - umask.flatten(), (loni, lati))
    pointsV = (mesh[VLON].values.flatten(), mesh[VLAT].values.flatten())
    vmaski = interp.griddata(pointsV, 1 - vmask.flatten(), (loni, lati))

    # Now processs files
    Ufiles = glob.glob(os.path.join(SRC_DIR,
                                    DATE,
                                    '*grid_U*.nc'))
    Ufiles.sort()
    
    Vfiles = glob.glob(os.path.join(SRC_DIR,
                                    DATE,
                                    '*grid_V*.nc'))
    Vfiles.sort()
    # open rotation pickle
    with open(rotation_file,'rb') as f:
        coeffs = pickle.load(f)
    
    for uf, vf in zip(Ufiles, Vfiles):
        bsU=os.path.basename(uf)
        bsV=os.path.basename(vf)
        newfile=os.path.join(OUT_DIR,
                             'interpolated_UV_{}.nc'.format(bsU.split('_')[6]))
        print('processing {} and {}'.format(bsU, bsV))
        if bsU.split('_')[6] != bsV.split('_')[6]:
            print('U/V files not on same date')
            exit()
        dU = xr.open_dataset(uf)
        uorig = dU[UVEL]
        dV = xr.open_dataset(vf)
        vorig = dV[VVEL]
        # rotate
        urot = rot_rep_2017_p(uorig, vorig, 'U','ij->e',coeffs)
        urot = np.ma.masked_array(urot.values, mask=np.ones(urot.values.shape) - umask)
        vrot = rot_rep_2017_p(uorig, vorig, 'V','ij->n',coeffs)
        vrot = np.ma.masked_array(vrot.values, mask=np.ones(vrot.values.shape) - vmask)
        # interpolate
        # U first
        uinterp = np.empty((urot.shape[0], loni.shape[0], loni.shape[1]))
        for t in range(uinterp.shape[0]):
            uinterp[t] = interp.griddata(pointsU, urot[t].flatten(), (loni, lati))
        uinterp = np.ma.masked_array(uinterp, mask=np.zeros(uinterp.shape) + umaski)
        # V next
        vinterp = np.empty((vrot.shape[0], loni.shape[0], loni.shape[1]))
        for t in range(vinterp.shape[0]):
            vinterp[t] = interp.griddata(pointsV, vrot[t].flatten(), (loni, lati))
        vinterp = np.ma.masked_array(vinterp, mask=np.zeros(vinterp.shape) + vmaski)
        
        # Prepare output
        dU = dU.drop(['time_instant', 'time_instant_bounds'])
        dU = dU.rename({'time_counter': 'time'})
        d = xr.Dataset(
            {'longitude': (['longitude'], loni[0, :]),
             'latitude': (['latitude'], lati[:, 0]),
             'u': (['time', 'latitude', 'longitude'], uinterp),
             'v': (['time', 'latitude', 'longitude'], vinterp)
        })
        d['time'] = dU.time
        d['latitude_longitude'] = xr.DataArray(
            0,
            attrs={'grid_mapping_name': 'latitude_longitude',
                   'longitude_of_prime_meridian': 0.,
                   'earth_radius': 6371229.0}) 
        d['u'].attrs = {'long_name': 'Zonal ocean surface current',
                        'standard_name': 'sea_water_x_velocity',
                        'short_name': 'u',
                        'units': 'm s-1',
                        'valid_min': -20.0,
                        'valid_max': 20.0}
        d['v'].attrs = {'long_name': 'Meridional ocean surface current',
                        'standard_name': 'sea_water_y_velocity',
                        'short_name': 'v',
                        'units': 'm s-1',
                        'valid_min': -20.0,
                        'valid_max': 20.0}
        d['latitude'].attrs = {'long_name': 'Latitude',
                               'standard_name': 'latitude',
                               'axis': 'Y',
                               'units': 'degrees_north',
                               'valid_min': -90.0,
                               'valid_max': 90.0}
        d['longitude'].attrs = {'long_name': 'Longitude',
                                'standard_name': 'longitude',
                                'axis': 'X',
                                'units': 'degrees_east',
                                'valid_min': -180.0,
                                'valid_max': 180.0}
        print('Saving in {}'.format(newfile))
        d.to_netcdf(newfile)


if __name__=='__main__':
    main()
