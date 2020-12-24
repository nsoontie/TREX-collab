# Script to interpolate HRDPS data to a regular latlon grid

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
import xarray as xr

SRC_DIR='/home/nso001/data/work2/models/hrdps-forecast/continuous/'
OUT_DIR='/home/nso001/data/work2/models/hrdps-forecast/continuous_interp/'

UVEL = 'u_wind'
VVEL = 'v_wind'
LAT = 'nav_lat'
LON = 'nav_lon'

DATE='*winds*y2020m09d2*'

# Set up new grid
LON_MIN, LON_MAX = -71, -56
LAT_MIN, LAT_MAX = 45, 51
DELTA=0.03 # lat/lon grid spacing for new grid

def main():
    # First, set up new grid
    loni = np.arange(LON_MIN, LON_MAX, DELTA)
    lati = np.arange(LAT_MIN, LAT_MAX, DELTA)
    loni, lati = np.meshgrid(loni, lati)

    # Now processs files
    files = glob.glob(os.path.join(SRC_DIR,
                                   DATE))
    files.sort()
      
    for f in files:
        bs=os.path.basename(f)
        newfile=os.path.join(OUT_DIR,
                             'interpolated_{}'.format(bs))
        print('processing {}'.format(bs))
        d = xr.open_dataset(f)
        uorig = d[UVEL]
        vorig = d[VVEL]
        lon = d[LON].values
        lon[lon>180] = lon[lon>180]-360
        lat = d[LAT].values
        points = (lon.flatten(), lat.flatten())
        # interpolate
        # U first
        uinterp = np.empty((uorig.values.shape[0], loni.shape[0], loni.shape[1]))
        for t in range(uinterp.shape[0]):
            uinterp[t] = interp.griddata(points, uorig.values[t].flatten(), (loni, lati))
        # V next
        vinterp = np.empty((vorig.values.shape[0], loni.shape[0], loni.shape[1]))
        for t in range(vinterp.shape[0]):
            vinterp[t] = interp.griddata(points, vorig.values[t].flatten(), (loni, lati))
        
        # Prepare output
        dold = d.rename({'time_counter': 'time'})
        dnew = xr.Dataset(
            {'longitude': (['longitude'], loni[0, :]),
             'latitude': (['latitude'], lati[:, 0]),
             'u_wind': (['time', 'latitude', 'longitude'], uinterp),
             'v_wind': (['time', 'latitude', 'longitude'], vinterp)
        })
        dnew['time'] = dold.time
        dnew['latitude_longitude'] = xr.DataArray(
            0,
            attrs={'grid_mapping_name': 'latitude_longitude',
                   'longitude_of_prime_meridian': 0.,
                   'earth_radius': 6371229.0}) 
        dnew['u_wind'].attrs = {'long_name': 'Zonal 10m wind',
                                 'standard_name': 'eastward_wind',
                                 'short_name': 'u_wind',
                                 'units': 'm s-1',
                                 'valid_min': -100.0,
                                 'valid_max': 100.0}
        dnew['v_wind'].attrs = {'long_name': 'Meridional 10m wind',
                                'standard_name': 'northward_wind',
                                'short_name': 'v_wind',
                                'units': 'm s-1',
                                'valid_min': -100.0,
                                'valid_max': 100.0}
        dnew['latitude'].attrs = {'long_name': 'Latitude',
                                  'standard_name': 'latitude',
                                  'axis': 'Y',
                                  'units': 'degrees_north',
                                  'valid_min': -90.0,
                                  'valid_max': 90.0}
        dnew['longitude'].attrs = {'long_name': 'Longitude',
                                   'standard_name': 'longitude',
                                   'axis': 'X',
                                   'units': 'degrees_east',
                                   'valid_min': -180.0,
                                   'valid_max': 180.0}
        print('Saving in {}'.format(newfile))
        dnew.to_netcdf(newfile)


if __name__=='__main__':
    main()
