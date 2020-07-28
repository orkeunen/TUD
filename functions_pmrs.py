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
from matplotlib import style




#own libary
import nsidc

#----------------------------



def loadCMSAV(points_xy, ds, windowlength, polyorder):
    
    df = pd.DataFrame()
    
    for n in [x for x in range(2)]: 
        x, y  = points_xy[n]
        C, M, ratio,  = nsidc.c_m_ratio(ds['TB'], x, y)
    
        df[n] = ratio
        dates = ratio.time.values
        df.index=dates
    
        df[n].fillna(method='bfill',inplace=True)
#         df[n][df[n]<0] = df[n].fillna(method='bfill',inplace=True)
#         df[n] = df[n].rolling(rollingvalue,center=True).median()  # apply and set the rolling median to rollingvalue
        
        df[n] = savgol_filter(df[n], windowlength, polyorder)

    df.columns = ['POI1', 'POI4']
    df.POI1.fillna(1, inplace=True) #make sure to remove the NaN values because the QR function does not work with them. Replace with a value to make sure the offset is still correct
    df.POI4.fillna(1, inplace=True)
    
    return df 



def loadCM(points_xy, ds, rollingvalue):
    
    df = pd.DataFrame()
    
    for n in [x for x in range(2)]: 
        x, y  = points_xy[n]
        C, M, ratio,  = nsidc.c_m_ratio(ds['TB'], x, y)
    
        df[n] = ratio
        dates = ratio.time.values
        df.index=dates
        
    
        df[n].fillna(method='bfill',inplace=True)
        df[n][df[n]<0] = df[n].fillna(method='bfill',inplace=True)
        df[n] = df[n].rolling(rollingvalue,center=True).median()  # apply and set the rolling median to rollingvalue
    df.columns = ['POI1', 'POI4']
    df.POI1.fillna(1, inplace=True) #make sure to remove the NaN values because the QR function does not work with them. Replace with a value to make sure the offset is still correct
    df.POI4.fillna(1, inplace=True)
    
    return df 
    
def applyshift(df,rollingvalue,shift,plot,SAT):
    
    df['POI1']=df['POI1'].shift(shift)
    
    if plot == 1:
        sns.set()
        sns.color_palette()
        style.use('seaborn-darkgrid')

        fig, ax = plt.subplots(figsize=(15,5))
        sns.set()
        
        ax.plot(df,marker='.', markersize=1.2,label= (f'CM ratio at POI - SG filter'))
        ax.set(xlabel=f'Years of {SAT} dataset', ylabel='CM Ratio (-)',
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
        sns.set()
        sns.color_palette()
        style.use('seaborn-darkgrid')
        
        fig, ax = plt.subplots(figsize=(15,5))
        sns.set()
        
        ax.plot(df_nqt ,marker='.', markersize=1.2,label= (f'CM ratio at POI - rolling median of {rollingvalue}'))
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
    quantiles = [0.10, 0.20, 0.30,0.40, 0.50, 0.60, 0.70, 0.80 , 0.90] #
    fits = [model.fit(q=q) for q in quantiles]

    x = df['POI1']
    y = df['POI4']

    fit = np.polyfit(x, y, deg=1)
    _x = np.linspace(x.min(), x.max(), num=len(y))

    res = model.fit(q=0.5)
    
    # fit lines
    _y_010 = fits[0].params['POI1'] * _x + fits[0].params['Intercept']
    _y_020 = fits[1].params['POI1'] * _x + fits[1].params['Intercept']
    _y_030 = fits[2].params['POI1'] * _x + fits[2].params['Intercept']
    _y_040 = fits[3].params['POI1'] * _x + fits[3].params['Intercept']
    _y_050 = fits[4].params['POI1'] * _x + fits[4].params['Intercept']
    _y_060 = fits[5].params['POI1'] * _x + fits[5].params['Intercept']
    _y_070 = fits[6].params['POI1'] * _x + fits[6].params['Intercept']
    _y_080 = fits[7].params['POI1'] * _x + fits[7].params['Intercept']
    _y_090 = fits[8].params['POI1'] * _x + fits[8].params['Intercept']

    # start and end coordinates of fit lines
    p = np.column_stack((x, y))
    a = np.array([_x[0], _y_010[0]]) #first point of 0.10 quantile fit line
    b = np.array([_x[-1], _y_010[-1]]) #last point of 0.10 quantile fit line

    a_ = np.array([_x[0], _y_090[0]])
    b_ = np.array([_x[-1], _y_090[-1]])

    a__ = np.array([_x[0], _y_020[0]]) #first point of 0.20 quantile fit line
    b__ = np.array([_x[-1], _y_020[-1]]) #last point of 0.20 quantile fit line

    a___ = np.array([_x[0], _y_080[0]])
    b___ = np.array([_x[-1], _y_080[-1]])

#mask based on if coordinates are above 0.95 or below 0.05 quantile fitlines using cross product

    mask = lambda p, a, b, a_, b_: (np.cross(p-a, b-a) > 0) | (np.cross(p-a_, b_-a_) < 0)
    mask = mask(p, a, b, a_, b_)
    
    if plot ==1:
        sns.set()
        sns.color_palette()
        style.use('seaborn-darkgrid')

#         print(res.summary())
        
        figure, axes = plt.subplots(figsize=(10,10))
        axes.scatter(x[mask], df['POI4'][mask], facecolor='r', edgecolor='none', alpha=0.3, label='data point outside outer quantiles')
        axes.scatter(x[~mask], df['POI4'][~mask], facecolor='g', edgecolor='none', alpha=0.3, label='data point inside outer quantiles')

        axes.plot(x, fit[0] * x + fit[1], label='best-fit', c='grey')
        axes.plot(_x, _y_090, label=quantiles[6], c='red')
        axes.plot(_x, _y_010, label=quantiles[0], c='red')
        axes.plot(_x, _y_080, label=quantiles[5], c='orange')
        axes.plot(_x, _y_020, label=quantiles[1], c='orange')

        axes.set_title(f'Quantile Regression function {location} given LT = {shift}')
        axes.legend(fancybox=True, framealpha=0.5)
        axes.set_xlabel('POI Upstream - CM RATIO')
        axes.set_ylabel('POI Downstream - CM RATIO')
        axes.text(1.09,1.02, f'R-squared = %0.3f' %res.prsquared,fontsize =14)
        axes.grid(b='bool')

# plt.savefig((f'/Users/oscarkeunen/Documents/1. TU Delft/1. Msc - Watermanagement/Afstuderen/satellite-cookbook-master/NSIDC-AMSRE/figures_python/Model Runs/{location}_LT({shift})_2QR.png'),dpi=600)

        plt.show()
    
    return (fits,res)

def probability(df, fits, plot, shift,location):
    
    x_set = df['POI1']

    _y_010 = fits[0].params['POI1'] * x_set + fits[0].params['Intercept']
    _y_020 = fits[1].params['POI1'] * x_set + fits[1].params['Intercept']
    _y_030 = fits[2].params['POI1'] * x_set + fits[2].params['Intercept']
    _y_040 = fits[3].params['POI1'] * x_set + fits[3].params['Intercept']
    _y_050 = fits[4].params['POI1'] * x_set + fits[4].params['Intercept']
    _y_060 = fits[5].params['POI1'] * x_set + fits[5].params['Intercept']
    _y_070 = fits[6].params['POI1'] * x_set + fits[6].params['Intercept']
    _y_080 = fits[7].params['POI1'] * x_set + fits[7].params['Intercept']
    _y_090 = fits[8].params['POI1'] * x_set + fits[8].params['Intercept']


    x_date = _y_050.reset_index().values[:,0]
    y10 = _y_010.reset_index().values[:,1].astype(float)
    y20 = _y_020.reset_index().values[:,1].astype(float)
    y30 = _y_030.reset_index().values[:,1].astype(float)
    y40 = _y_040.reset_index().values[:,1].astype(float)
    y50 = _y_050.reset_index().values[:,1].astype(float)
    y60 = _y_060.reset_index().values[:,1].astype(float)
    y70 = _y_070.reset_index().values[:,1].astype(float)
    y80 = _y_080.reset_index().values[:,1].astype(float)
    y90 = _y_090.reset_index().values[:,1].astype(float)

    y10 = np.array([y10[i] for i in range (0,len(y10))])
    y20 = np.array([y20[i] for i in range (0,len(y20))])
    y30 = np.array([y30[i] for i in range (0,len(y30))])
    y40 = np.array([y40[i] for i in range (0,len(y40))])
    y50 = np.array([y50[i] for i in range (0,len(y50))])
    y60 = np.array([y60[i] for i in range (0,len(y60))])
    y70 = np.array([y70[i] for i in range (0,len(y70))])
    y80 = np.array([y80[i] for i in range (0,len(y80))])
    y90 = np.array([y90[i] for i in range (0,len(y90))])

    if plot == 1:
        sns.set()
        sns.color_palette()
        style.use('seaborn-darkgrid')
        


        fig, ax = plt.subplots(figsize=(17,7))
        ax.fill_between(x_date, y10, y20, alpha =0.3,linewidth =0.0, color='tab:grey',label='<20% probability function')
        ax.fill_between(x_date, y20, y80, alpha =0.6,linewidth =0.0, color='tab:grey',label='80%-20% probability function')
        ax.fill_between(x_date, y80, y90, alpha =0.3,linewidth =0.0 ,color='tab:grey',label='>20% probability function')
        ax.plot(x_date,y50,'r',alpha =1,linestyle='dashed',label='50% probability function')
        ax.set(xlabel='Years of ASMR-E dataset', ylabel='CM Ratio (-)',
               title=f'{location} - Probability of Exceedance given LT = {shift}')
        ax.legend(fancybox=True, framealpha=0.5)
        ax.tick_params(labelsize=12)
        ax.grid(b=bool)

    return(y10,y20,y30,y40,y50,y60,y70,y80,y90)
    
    
    



def calc_performance_scores_new(df_nqt, obs, pred, threshold , dt, percentile):

    df_nqt = np.where((df_nqt.index.month == 2) & (df_nqt.index.day == 1))[0] # select only the moment in time the rainseason starts to find the first moment above the trheshold
    performance = np.zeros((len(df_nqt),4)) # create performance matrix 1 = date obs_threshold 2 obs_threshold 3. date pred_thres 4 pred_thres
   
    for t in range (30): # 30 for DSMP data set, 8 for AMSR-E, (4,5 for a year)
        obs_threshold = np.where((obs[df_nqt[t]: df_nqt[t]+365]) > threshold)[0] 


        if len(obs_threshold) > 0:   #alles wat groter dan nul is -> hit of een miss
                obs_threshold0 = obs_threshold[0] + df_nqt[t]
                pred_threshold = np.where(pred[percentile][obs_threshold0-dt:obs_threshold0+dt] > threshold)[0] + (obs_threshold0-dt)
                
                if len(pred_threshold) > 0: #hit
                    performance [t][0] = obs_threshold0 
                    performance [t][1] = 1
                    performance [t][2] = min(pred_threshold)       
                    performance [t][3] = 1      
#                     print('hit')
                else:       # miss!            
                    performance [t][0] = obs_threshold0 
                    performance [t][1] = 1           
                    performance [t][2] = len(pred_threshold)
                    performance [t][3] = 0
#                     print('Miss')       
                    
      
        
        pred_threshold2 = np.where((pred[percentile][df_nqt[t]: df_nqt[t]+365]) > threshold)[0]

        if len(pred_threshold2)> 0: #alles wat groter dan nul is ->  FA of HIT
                
                pred_threshold3 = pred_threshold2[0]  + df_nqt[t]
                obs_threshold2 = np.where(obs[pred_threshold3-dt:pred_threshold3+dt] > threshold)[0]
               
                if len(obs_threshold2) == 0: #alles wat groter dan nul is -> FA of CN
                    
                    performance [t][0] = len(obs_threshold2) 
                    performance [t][1] = 0
                    performance [t][2] = pred_threshold3
                    performance [t][3] = 1    
#                     print('false alarm')   
       
        if len(obs_threshold) == 0: # CN
             if len(pred_threshold2) == 0:
                    
                    performance [t][0] = -999
                    performance [t][1] = 0           
                    performance [t][2] = -999
                    performance [t][3] = 0     
#                     print('corr neg')

    performance =pd.DataFrame(performance)
    
    performance.columns = ['day obs', 'obs', 'day pred', 'pred']    
    hits = len(np.where((performance.obs==1) & (performance.pred ==1))[0])    
    false_al = len(np.where((performance.obs==0) & (performance.pred ==1))[0])
    misses = len(np.where((performance.obs==1) & (performance.pred ==0))[0])
    corr_neg = len(np.where((performance.obs==0) & (performance.pred ==0))[0])



    try:
        output = np.zeros((4,))
        
        if hits+misses ==0:
            output[0]=np.nan
        else:
            output[0] = hits / (hits + misses) #Probability of Detection
       
        if hits+false_al ==0:
            output[1]=np.nan
        else:
            output[1] = false_al / (hits + false_al) #False Alarm Rate
        
        if false_al+corr_neg ==0:
            output[2]=np.nan
        else:
            output[2] = false_al / (false_al + corr_neg) #Probability of fase detection

        if hits+false_al+misses ==0:
            output[3]=np.nan
        else:
            output[3] = hits / (hits + false_al + misses) #Critical succes index

    except ZeroDivisionError:
        return -99
   

    metric = np.zeros((4,))
    metric[0] = hits
    metric[1] = false_al
    metric[2] = misses
    metric[3] = corr_neg

    
    return (performance,output,metric)
      
    
        
    
    
    
    
#     