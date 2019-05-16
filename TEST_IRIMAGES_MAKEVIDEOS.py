# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 21:38:08 2018

@author: z3439910
"""

import numpy as np
import xarray as xr
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import time
import h5py

from scipy import ndimage
from matplotlib import colors
from skimage.morphology import watershed
from skimage import measure
from skimage.feature import peak_local_max
from skimage.morphology import reconstruction
import glob

#WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project"
WORKPLACE = r"K:"
IRDIR = WORKPLACE + r"\IRimages2017"
SAVDIR = IRDIR + r"\Videos"
BTDIR = WORKPLACE + r"\2_IBTrACSfiles"
os.chdir(IRDIR)

    #%% Acquire BT images
#BTemp_filename = r"merg_2012071811_4km-pixel.nc4"
files = glob.glob("*.nc4")

I_minute = 30
for file in files:
    I_minute = 0
    filename = file[5:15] + "00"
    
    if I_minute == 0:
        #slice out BT images for the current basin
        C_BTemp = xr.open_dataset(IRDIR+ "\\" + file)['Tb'].values[0]
        C_lat = xr.open_dataset(IRDIR+ "\\" + file)['lat'].values
        C_lon = xr.open_dataset(IRDIR+ "\\" + file)['lon'].values
        #interpolate NaN values in BT images
    #    mask = np.isnan(C_BTemp)
    #    C_BTemp[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), C_BTemp[~mask])
      
    elif I_minute == 30:
        #slice out BT images for the current basin
        C_BTemp = xr.open_dataset(IRDIR+ "\\" + file)['Tb'].values[1]
        C_lat = xr.open_dataset(IRDIR+ "\\" + file)['lat'].values
        C_lon = xr.open_dataset(IRDIR+ "\\" + file)['lon'].values
    #    #interpolate NaN values in BT images
    #    mask = np.isnan(C_BTemp)
    #    C_BTemp[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), C_BTemp[~mask])
    #
    
    fig = plt.figure()
    lat_max = np.round(np.max(C_lat),1)
    lat_min = np.round(np.min(C_lat),1)
    lon_max = np.round(np.max(C_lon),1)
    lon_min = np.round(np.min(C_lon),1)
    im = plt.imshow(C_BTemp, extent = (lon_min, lon_max, lat_min, lat_max), cmap='Greys',origin='lower')
    #plt.plot(I_lon[C_i],I_lat[C_i],'or', markersize = 2) 
    ax = plt.gca()
    ax.set_title(filename)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.savefig(SAVDIR + "\\" + filename +".png",dpi=200)
    plt.close()
    
    I_minute = 30
    filename = file[5:15] + "30"
    
    if I_minute == 0:
        #slice out BT images for the current basin
        C_BTemp = xr.open_dataset(IRDIR+ "\\" + file)['Tb'].values[0]
        C_lat = xr.open_dataset(IRDIR+ "\\" + file)['lat'].values
        C_lon = xr.open_dataset(IRDIR+ "\\" + file)['lon'].values
        #interpolate NaN values in BT images
    #    mask = np.isnan(C_BTemp)
    #    C_BTemp[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), C_BTemp[~mask])
      
    elif I_minute == 30:
        #slice out BT images for the current basin
        C_BTemp = xr.open_dataset(IRDIR+ "\\" + file)['Tb'].values[1]
        C_lat = xr.open_dataset(IRDIR+ "\\" + file)['lat'].values
        C_lon = xr.open_dataset(IRDIR+ "\\" + file)['lon'].values
    #    #interpolate NaN values in BT images
    #    mask = np.isnan(C_BTemp)
    #    C_BTemp[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), C_BTemp[~mask])
    #
    
    fig = plt.figure()
    lat_max = np.round(np.max(C_lat),1)
    lat_min = np.round(np.min(C_lat),1)
    lon_max = np.round(np.max(C_lon),1)
    lon_min = np.round(np.min(C_lon),1)
    im = plt.imshow(C_BTemp, extent = (lon_min, lon_max, lat_min, lat_max), cmap='Greys',origin='lower')
    #plt.plot(I_lon[C_i],I_lat[C_i],'or', markersize = 2) 
    ax = plt.gca()
    ax.set_title(filename)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.savefig(SAVDIR + "\\" + filename +".png",dpi=200)
    plt.close()
    