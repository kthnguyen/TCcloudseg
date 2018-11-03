# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 08:41:38 2018

@author: z3439910
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import time
from matplotlib import colors
import h5py
from scipy import ndimage
import os
import cv2

WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project"
IRDIR = WORKPLACE + r"\IRimages2012"
BTDIR = WORKPLACE + r"\2_IBTrACSfiles"
RADDIR = WORKPLACE+ r"\6_CERESdata"

TC_serial = '2012140N33283'

LAT_BOUND = [-20,60] #NA Basin
LON_BOUND = [-120,0] #NA Basin
#%
def get_BTempimage_bound(latmin,latmax,lonmin,lonmax):
    BTempimage = xr.open_dataset(WORKPLACE+ "\IRimages2012\merg_2012080100_4km-pixel.nc4")
    BTemp_lat = BTempimage['lat'].values[:]
    BTemp_lon = BTempimage['lon'].values
    lat_bound = [i for i,val in enumerate(BTemp_lat) if (val>latmin and val<latmax)]   
    lat_val_bound = [val for i,val in enumerate(BTemp_lat) if (val>latmin and val<latmax)] 
    lon_bound = [i for i,val in enumerate(BTemp_lon) if (val>lonmin and val<lonmax)]   
    lon_val_bound = [val for i,val in enumerate(BTemp_lon) if (val>lonmin and val<lonmax)] 
    return[lat_bound[0],lat_bound[-1],lon_bound[0],lon_bound[-1]]
#% Get idices in accordance with brightness temperature images

DIM_BOUND = get_BTempimage_bound(LAT_BOUND[0],LAT_BOUND[1],LON_BOUND[0],LON_BOUND[1])#incices from BT images

B_tracks = xr.open_dataset(BTDIR+"\\"+"Year.2012.ibtracs_all.v03r10.nc")

B_TC_serials = B_tracks['storm_sn'].values
B_TC_names = B_tracks['name'].values


for i,j in enumerate(B_TC_serials):
    if j.decode("utf-8") == TC_serial:
        I_TC_idx = i
## extract variables into arrays
I_name = B_TC_names[I_TC_idx].decode("utf-8")
I_TC_time = B_tracks['source_time'].values[I_TC_idx,:]
I_TC_time = pd.DataFrame(I_TC_time).dropna().values[:,0]


I_lat = B_tracks['lat_for_mapping'].values[I_TC_idx,:]
I_lat = pd.DataFrame(I_lat).dropna().values[:,0]
I_lon = B_tracks['lon_for_mapping'].values[I_TC_idx,:]
I_lon = pd.DataFrame(I_lon).dropna().values[:,0]

# interpolate best track lat long to 0.5-hour intervals
df = pd.DataFrame({'time':I_TC_time,'lat':I_lat,'lon':I_lon})
df = df.set_index('time')
df_reindexed = df.reindex(pd.date_range(start=I_TC_time[0],end=I_TC_time[len(I_TC_time)-1],freq='0.5H'))
I_time_interpolate = df_reindexed.interpolate(method='time')
I_time_interpolate.index.name = 'time'
I_time_interpolate.reset_index(inplace = True)
I_year = pd.to_datetime(I_time_interpolate['time'].values).year
I_month = pd.to_datetime(I_time_interpolate['time'].values).month
I_day = pd.to_datetime(I_time_interpolate['time'].values).day
I_hour = pd.to_datetime(I_time_interpolate['time'].values).hour
I_minute = pd.to_datetime(I_time_interpolate['time'].values).minute
I_lat = I_time_interpolate['lat']
I_lon = I_time_interpolate['lon']
#% Create an HDF5 file to store label for the current storm
DIM_LAT = DIM_BOUND[1]-DIM_BOUND[0] + 1
DIM_LON = DIM_BOUND[3]-DIM_BOUND[2] + 1
DIM_TIME = np.shape(I_time_interpolate['time'])[0]
#%%
SAVDIR = WORKPLACE + r"\3_Figures\\" + TC_serial + "_" + I_name
HFILE_DIR = SAVDIR + r"\\" + TC_serial + r"_" + I_name + r'_labelsa.h5'
Hfile_label = h5py.File(HFILE_DIR)  
C_label_TC = Hfile_label['label_TC']
C_label_BG = Hfile_label['label_BG']

HFILE_DIR2 = SAVDIR + r"\\" + TC_serial + r"_" + I_name + r'_labels.h5'
Hfile_label2 = h5py.File(HFILE_DIR2,'w')
Hfile_label2.close()

Hfile_label2 = h5py.File(HFILE_DIR2,'r+')
Hfile_label2.create_dataset('label_TC', shape = (DIM_TIME,DIM_LAT,DIM_LON),chunks=True)

Hfile_label2.close()

#% Start spreading 
# open the label HDF5 file
Hfile_label2 = h5py.File(HFILE_DIR2)  

h5py.h5o.copy(Hfile_label.id,"label_BG",Hfile_label2.id,"label_TC")
Hfile_label2.close()
Hfile_label.close()

#%%
Hfile_label2 = h5py.File(HFILE_DIR+r".h5",'r+')  
C_label_TC2 = Hfile_label['label_TC']
#%%
a = C_label_TC[1,:,:]
