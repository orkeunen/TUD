import pyproj
import xarray as xr
import numpy as np
import os
import subprocess
import shutil
import datetime


def proj_coord(coord, proj_in, proj_out):
    """
    Returns a packaged tuple (x, y) coordinate in projection proj_out
    from one packaged tuple (x, y) coordinat ein projection proj_in
    Inputs:
        coord: tuple (x, y)
        proj_in: pyproj.Proj format projection
        proj_out: pyproj.Proj format projection
    Outputs:
        tuple (x, y)
        
    """
    x, y = coord
    return pyproj.transform(proj_in, proj_out, x, y)

def proj_coords(coords, proj_in, proj_out):
    """
    project a list of coordinates, return a list.
    Inputs:
        coords: list of tuples (x, y)
        proj_in: pyproj.Proj format projection
        proj_out: pyproj.Proj format projection
    Outputs:
        list of tuples (x, y)
    
    """ 
    return [proj_coord(coord, proj_in, proj_out) for coord in coords]

def select_bounds(ds, bounds):
    """
    selects xarray ds along a provided bounding box
    assuming slicing should be done over coordinate axes x and y (hard coded, I was lazy....:-()
    """
    
    xs = slice(bounds[0][0], bounds[1][0])
    ys = slice(bounds[1][1], bounds[0][1])
    # select over x and y axis
    return ds.sel(x=xs, y=ys)

def make_measures_url(date, res, freq, HV, AD):
    """
    Prepares a url for Measures data to download.
    url_template - str url with placeholders for date (%Y.%m.%d), resolution (:d, km), date (%Y%j),
        frequency (str), polarisation ('H'/'V'), ascending/descending path ('A', 'D')
    
    """
    url_base = 'https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0630.001'
    url_folder = '{:s}'
    url_template = os.path.join(url_base, url_folder,
                                'NSIDC-0630-EASE2_T{:s}km-AQUA_AMSRE-{:s}-{:s}{:s}-{:s}-{:s}-v1.3.nc') 

    datestr1 = date.strftime('%Y%j')
    datestr2 = date.strftime('%Y.%m.%d')
#     if str(res) == '3.125':
#         suffix = 'SIR-RSS'
    if str(res) == '25':
        suffix = 'GRD-RSS'
    #else:
        #suffix = 'SIR-CSU'
   # else:
    #    suffix = 'SIR-RSS'
    
    return url_template.format(datestr2, str(res), datestr1, freq, HV, AD, suffix)

def make_measures_download(url, username, password):
    download_template = 'wget --http-user={:s} --http-password={:s} --load-cookies mycookies.txt --save-cookies mycookies.txt --keep-session-cookies --no-check-certificate --auth-no-challenge -r --reject "index.html*" -np -e robots=off {:s}'
    return download_template.format(username, password, url)

def download_measures(freq, res, HV, AD, date, username, password):
    url = make_measures_url(date, res, freq, HV, AD)
    download_string = make_measures_download(url, username, password)
    return url, subprocess.call(download_string.split(' '))  # call the download string in a command-line (use wget.exe! get it from online)

def download_measures_ts(freq, res, HV, AD, start_date, end_date, bounds, fn_out_prefix, username, password):
    """
    Downloads and slices in space, a series of NSIDC daily files, conditioned on user inputs
    """
    # convert bounds to projected coordinates
    step = datetime.timedelta(days=1)  # M

    proj4str = '+proj=cea +lat_0=0 lon_0=0 +lat_ts=30 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m'
    proj_out = pyproj.Proj('epsg:4326')
    # we define a projection object for the projection used in the downloaded grids.
    proj_in = pyproj.Proj(proj4str)

    # here we convert the coordinates in lat-lon into the coordinate system of the downloaded grids.
    bounds_xy = proj_coords(bounds, proj_out, proj_in)
    # points_xy = proj_coords(points_interest, proj_out, proj_in)

    date = start_date
    list_ds = []
    year = date.year  # let's store data per year
    while date <= end_date:
        url, success = download_measures(freq, res, HV, AD, date, username, password)
        fn = url.strip('https://')  # strip https:// from the url to get the local location of the downloaded file
        path = fn.split('/')[0]  # split fn in little pieces on / and keep only the 0-th indexed piece (main folder)
        if success == 0:
            print('Retrieved {:s}'.format(url))
            # the file was successfully downloaded (if not zero, there was a problem or file is simply not available)
            # read file, cut a piece and add it to our list of time steps
            ds = xr.open_dataset(fn, decode_cf=False)
            ds_sel = select_bounds(ds, bounds_xy)
            list_ds.append(ds_sel.load())  # load the actual data so that we can delete the original downloaded files
            ds.close()
            shutil.rmtree(path)  # we're done cutting part of the required grid, so throw away the originally downloaded world grid.
        date += step   # increase by one day to go for the next download day.
        if (year != date.year) or (date > end_date):  # store results if one moves to a new year or the end date is reached
            # concatenate the list of timesteps into a new ds
            if len(list_ds) > 0:
                # only store if any values were found
                ds_year = correct_miss_fill(xr.concat(list_ds, dim='time'))
                # ds_year.data_vars
                encoding = {var: {'zlib': True} for var in ds_year.data_vars if var != 'crs'}
                # store the current dataset into a nice netcdf file
                # fn_out = os.path.abspath(os.path.join(out_path, fn_out_template.format(str(res), freq, HV, AD, year)))
                fn_out = fn_out_prefix + '_' + str(year) + '.nc'
                print('Writing output for year {:d} to {:s}'.format(year, fn_out))
                ds_year.to_netcdf(fn_out, encoding=encoding)
            # prepare a new dataset
            list_ds = []  # empty list
            year = date.year  # update the year

def plot_points(ax, points, **kwargs):
    x_point, y_point = zip(*points)
    ax.plot(x_point, y_point, **kwargs)
    return ax

def correct_miss_fill(ds):
    """
    Returns a properly decoded Climate-and-Forecast conventional ds, after correction of a conflicting attribute (sjeez....)
    """
    for d in ds.data_vars:
        try:
            ds[d].attrs.update({'missing_value': ds[d]._FillValue})
        except:
            pass
    return xr.decode_cf(ds)

def c_m_ratio(ds_tb, x, y, x_off=62500, y_off=62500):
    def cc(ts1, M):
        coef = np.ma.corrcoef(np.ma.masked_invalid(ts1.values.flatten()), np.ma.masked_invalid(M.values.flatten()))[1][0]
        return xr.DataArray(coef)
    ds_tb_sel = ds_tb.sel(x=slice(x-x_off, x+x_off), y=slice(y+y_off, y-y_off))
    # convert the xarray data-array into a bunch of point time series
    # select series in (M)easurement location
    M = ds_tb_sel.sel(x=x, y=y, method='nearest')
    tb_points = ds_tb_sel.stack(points=('y', 'x')) # .reset_index(['x', 'y'], drop=True) # .transpose('points', 'time')
    # add a coordinate axis to the points
    # apply the function over all points to calculate the trend at each point
#     import pdb;pdb.set_trace()
    coefs = tb_points.groupby('points').apply(cc, M=M)
    # unstack back to lat lon coordinates
    coefs_2d = coefs.unstack('points').rename(dict(points_level_0='y', points_level_1='x')) # get the 2d back and rename axes back to x, y
    #LOWEST CALIBRATION
     # find the x/y index where the correlation is lowest
    idx_y, idx_x = np.where(coefs_2d==coefs_2d.min())
     # select  series in (C)alibration location (with lowest correlation)
        # iport pdb;pdb.set_trace()
    C = ds_tb_sel[:, idx_y, idx_x].squeeze(['x', 'y']).drop(['x', 'y'])  # get rid of the x and y coordinates of calibration pixel
#     # which has the lowest correlation with the point of interest?
    ratio = C / M
#     ratio_mean = C_mean / M
    return C, M, ratio ,# C_mean, ratio_mean
