import pyproj
import xarray as xr
import numpy as np

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

def make_measures_url(url_template, res, dt, freq, HV, AD):
    """
    Prepares a url for Measures data to download.
    url_template - str url with placeholders for date (%Y.%m.%d), resolution (:d, km), date (%Y%j),
        frequency (str), polarisation ('H'/'V'), ascending/descending path ('A', 'D')
    
    """
    datestr1 = dt.strftime('%Y%j')
    datestr2 = dt.strftime('%Y.%m.%d')
#     if str(res) == '25':
      if str(res) == '3.125':
        suffix = 'SIR-RSS' #suffix for the 3.125 km res
#         suffix = 'GRD-CSU'

    
    return url_template.format(datestr2, str(res), datestr1, freq, HV, AD, suffix)

def make_measures_download(download_template, url, username, password):
    return download_template.format(username, password, url)

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
    C = ds_tb_sel[:, idx_y, idx_x].squeeze(['x', 'y']).drop(['x', 'y'])  # get rid of the x and y coordinates of calibration pixel
    # which has the lowest correlation with the point of interest?
    ratio = C / M
    # #MEAN CALIBRATION
    # # find the x/y index where the correlation is mean
    # idx_y_mean, idx_x_mean = np.where(coefs_2d==coefs_2d.mean())
    # # select  series in (C)alibration location (with lowest correlation)
    # C_mean = ds_tb_sel[:, idx_y_mean, idx_x_mean].squeeze(['x','y']).drop(['x','y'])  # get rid of the x and y coordinates of calibration pixel
    # ratio_mean = C_mean / M
    return C, M, ratio #, C_mean, ratio_mean
