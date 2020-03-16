#1. IMPORTING PACKAGES AND LIBRARIES


import os
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.img_tiles import OSM
from cartopy.io.img_tiles import GoogleTiles as moa
from cycler import cycler
import scipy
import seaborn as sns
import glob
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xarray as xr
import pyproj
import osr
import datetime
import subprocess
import shutil
import numpy as np
from ipywidgets import interact
import pandas as pd
import pickle
from scipy.signal import savgol_filter
import time
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot
from sklearn import preprocessing



#own libary
import nsidc

#----------------------------



def loadCM(points_xy,ds, rollingvalue):
    
    df = pd.DataFrame()
    
    for n in [x for x in range(2)]: 
        x, y  = points_xy[n]
        C, M, ratio,  = nsidc.c_m_ratio(ds['TB'], x, y)
    
        df[n] = ratio
        dates = ratio.time.values
        df.index=dates
    
        df[n].fillna(method='bfill',inplace=True)
        df[n][df[n]<0] = df[n].fillna(method='bfill',inplace=True)
        df[n] = df[n].rolling(rollingvalue,center=False).median()  # apply and set the rolling median to rollingvalue
    df.columns = ['POI1', 'POI4']
    df.POI1.fillna(1, inplace=True) #make sure to remove the NaN values because the QR function does not work with them. Replace with a value to make sure the offset is still correct
    df.POI4.fillna(1, inplace=True)
    
    return df 
    
def applyshift(df,rollingvalue,shift,plot):
    
    df['POI1']=df['POI1'].shift(shift)
    
    if plot == 1:
        
        fig, ax = plt.subplots(figsize=(15,5))
        sns.set()
        
        ax.plot(df,marker='.', markersize=1.2,label= (f'CM ratio at POI - rolling median of {rollingvalue}'))
        ax.set(xlabel='Years of ASMR-E dataset', ylabel='CM Ratio (-)',
           title=f' CM Ratio at the selected locations for Senanga given LT = {shift}')
        ax.legend()
        ax.tick_params(labelsize=12)
        ax.grid(b=bool)
        
#         trigger = df.quantile([0.9, 0.95, 0.98])

        
#         plt.axhline(y=trigger[0][0.9],linewidth=0.8, color='r',label = 'trigger 90% percentile')    
#         plt.axhline(y=trigger[][0.95],linewidth=0.8, color='r',label = 'trigger 95% percentile')
#         plt.axhline(y=trigger[2][0.98],linewidth=0.8, color='r',label = 'trigger 98% percentile')
    
    return df

def NQT(df,plot, shift, location, rollingvalue):
    
    df_temp = np.sort(df,axis=1)
    quantile_transformer = preprocessing.QuantileTransformer(output_distribution='uniform',copy=True)


    X_trans = quantile_transformer.fit_transform(df_temp)
    X_trans = quantile_transformer.inverse_transform(X_trans)


    df_nqt =pd.DataFrame({'POI1': X_trans[:, 0], 'POI4': X_trans[:, 1]},index =df.index) 
      
        
    if plot ==1:       
        
        fig, ax = plt.subplots(figsize=(15,5))
        sns.set()
        
        ax.plot(df_nqt ,marker='.', markersize=1.2,label= (f'CM ratio at POIs - rolling median of {rollingvalue}'))
        ax.set(xlabel='Years of ASMR-E dataset', ylabel='CM Ratio (-)',
           title=f' CM Ratio at the selected locations for Senanga given LT = {shift}')
        ax.legend()
        ax.tick_params(labelsize=12)
        ax.grid(b=bool)
    
    return (df_nqt)

def calculateQR(df,selection,plot,shift,location):
    # QR plot of the Dataframe
    df.POI1.fillna(1, inplace=True)
    df.POI4.fillna(1, inplace=True)

    x_set = df['POI1']

    df= df[df.index.month<selection] #select only the months in the rainy season to look at the effects of the CM ratio for inundation. 

    model = smf.quantreg('POI4 ~ POI1', df) #call the model
    quantiles = [0.10, 0.25, 0.75 , 0.90, 0.50]
    fits = [model.fit(q=q) for q in quantiles]

    x = df['POI1']
    y = df['POI4']

    fit = np.polyfit(x, y, deg=1)
    _x = np.linspace(x.min(), x.max(), num=len(y))

    res = model.fit(q=0.5)
    
    # fit lines
    _y_010 = fits[0].params['POI1'] * _x + fits[0].params['Intercept']
    _y_090 = fits[3].params['POI1'] * _x + fits[3].params['Intercept']
    _y_025 = fits[1].params['POI1'] * _x + fits[1].params['Intercept']
    _y_075 = fits[2].params['POI1'] * _x + fits[2].params['Intercept']
    _y_050 = fits[4].params['POI1'] * _x + fits[4].params['Intercept']


    # start and end coordinates of fit lines
    p = np.column_stack((x, y))
    a = np.array([_x[0], _y_010[0]]) #first point of 0.05 quantile fit line
    b = np.array([_x[-1], _y_010[-1]]) #last point of 0.05 quantile fit line

    a_ = np.array([_x[0], _y_090[0]])
    b_ = np.array([_x[-1], _y_090[-1]])

    a__ = np.array([_x[0], _y_025[0]]) #first point of 0.10 quantile fit line
    b__ = np.array([_x[-1], _y_025[-1]]) #last point of 0.10 quantile fit line

    a___ = np.array([_x[0], _y_075[0]])
    b___ = np.array([_x[-1], _y_075[-1]])

#mask based on if coordinates are above 0.95 or below 0.05 quantile fitlines using cross product

    mask = lambda p, a, b, a_, b_: (np.cross(p-a, b-a) > 0) | (np.cross(p-a_, b_-a_) < 0)
    mask = mask(p, a, b, a_, b_)
    
    if plot ==1:
        print(res.summary())
        
        figure, axes = plt.subplots(figsize=(10,10))
        axes.scatter(x[mask], df['POI4'][mask], facecolor='r', edgecolor='none', alpha=0.3, label='data point outside outer quantiles')
        axes.scatter(x[~mask], df['POI4'][~mask], facecolor='g', edgecolor='none', alpha=0.3, label='data point inside outer quantiles')

        axes.plot(x, fit[0] * x + fit[1], label='best-fit', c='grey')
        axes.plot(_x, _y_090, label=quantiles[0], c='red')
        axes.plot(_x, _y_010, label=quantiles[3], c='red')
        axes.plot(_x, _y_075, label=quantiles[1], c='orange')
        axes.plot(_x, _y_025, label=quantiles[2], c='orange')

        axes.set_title(f'Quantile Regression function {location} given LT = {shift}')
        axes.legend(fancybox=True, framealpha=0.5)
        axes.set_xlabel('POI1 - CM RATIO')
        axes.set_ylabel('POI4 - CM RATIO')
        axes.text(0.97,0.97, f'R-squared = %0.3f' %res.prsquared,fontsize =14)
        axes.grid(b='bool')

# plt.savefig((f'/Users/oscarkeunen/Documents/1. TU Delft/1. Msc - Watermanagement/Afstuderen/satellite-cookbook-master/NSIDC-AMSRE/figures_python/Model Runs/{location}_LT({shift})_2QR.png'),dpi=600)

        plt.show()
    
    return (fits,res)

def probability(df, fits, plot, shift,location):
    
    x_set = df['POI1']
    
    _y_010 = fits[0].params['POI1'] * x_set + fits[0].params['Intercept']
    _y_090 = fits[3].params['POI1'] * x_set + fits[3].params['Intercept']
    _y_025 = fits[1].params['POI1'] * x_set + fits[1].params['Intercept']
    _y_075 = fits[2].params['POI1'] * x_set + fits[2].params['Intercept']
    _y_050 = fits[4].params['POI1'] * x_set + fits[4].params['Intercept']

    x_date = _y_050.reset_index().values[:,0]
    y50 = _y_050.reset_index().values[:,1].astype(float)
    y25 = _y_025.reset_index().values[:,1].astype(float)
    y10 = _y_010.reset_index().values[:,1].astype(float)
    y75 = _y_075.reset_index().values[:,1].astype(float)
    y90 = _y_090.reset_index().values[:,1].astype(float)

    y50 = np.array([y50[i] for i in range (0,len(y50))])
    y25 = np.array([y25[i] for i in range (0,len(y25))])
    y10 = np.array([y10[i] for i in range (0,len(y10))])
    y75 = np.array([y75[i] for i in range (0,len(y75))])
    y90 = np.array([y90[i] for i in range (0,len(y90))])

    if plot == 1:

        fig, ax = plt.subplots(figsize=(17,7))
        ax.fill_between(x_date, y10, y25, alpha =0.3,linewidth =0.0, color='tab:grey',label='95% probability function')
        ax.fill_between(x_date, y25, y75, alpha =0.6,linewidth =0.0, color='tab:grey',label='90%-10% probability function')
        ax.fill_between(x_date, y75, y90, alpha =0.3,linewidth =0.0 ,color='tab:grey',label='5% probability function')
        ax.plot(x_date,y50,'r',alpha =1,linestyle='dashed',label='50% probability function')
        ax.set(xlabel='Years of ASMR-E dataset', ylabel='CM Ratio (-)',
               title=f'{location} - Probability of Exceedance given LT = {shift}')
        ax.legend(fancybox=True, framealpha=0.5)
        ax.tick_params(labelsize=12)
        ax.grid(b=bool)

    return(y10,y25,y50,y75,y90)
    
    
    
    
    
    
    
    
    