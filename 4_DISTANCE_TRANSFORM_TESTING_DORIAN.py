# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 21:38:08 2018

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
from skimage.morphology import watershed

WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project"
IRDIR = WORKPLACE + r"\IRimages2013"
SAVDIR = WORKPLACE + r"\3_Figures"
BTDIR = WORKPLACE + r"\2_IBTrACSfiles"
os.chdir(IRDIR)

#% Functions
def calcdistance_km(latA,lonA,latB,lonB):
    dist = np.sqrt(np.square(latA-latB)+np.square(lonA-lonB))*111
    return np.int(dist)
#    return True\
    
#%
def time_to_string_with_min(iyear, imonth, iday, ihour, iminute):   
    str_iyear = str(iyear)
    if imonth < 10:
        str_imonth = "0" + str(imonth)
    else:
        str_imonth = str(imonth)
    
    if iday < 10:
        str_iday = "0" + str(iday)
    else:
        str_iday = str(iday)      
    
    if ihour < 10:
        str_ihour = "0" + str(ihour)
    else:
        str_ihour = str(ihour) 
        
    if iminute < 10:
        str_iminute = "0" + str(iminute)
    else:
        str_iminute = str(iminute)
    
    str_itime = str_iyear + str_imonth + str_iday + str_ihour + str_iminute
    return str_itime

#%
def time_to_string_without_min(iyear, imonth, iday, ihour):   
    str_iyear = str(iyear)
    if imonth < 10:
        str_imonth = "0" + str(imonth)
    else:
        str_imonth = str(imonth)
    
    if iday < 10:
        str_iday = "0" + str(iday)
    else:
        str_iday = str(iday)      
    
    if ihour < 10:
        str_ihour = "0" + str(ihour)
    else:
        str_ihour = str(ihour) 
    
    str_itime = str_iyear + str_imonth + str_iday + str_ihour
    return str_itime

#%
def get_BTempimage_bound(latmin,latmax,lonmin,lonmax):
    BTempimage = xr.open_dataset(WORKPLACE+ "\IRimages2012\merg_2012080100_4km-pixel.nc4")
    latmin = 0
    latmax = 60
    BTemp_lat = BTempimage['lat'].values[:]
    BTemp_lon = BTempimage['lon'].values
    lat_bound = [i for i,val in enumerate(BTemp_lat) if (val>latmin and val<latmax)]   
    lat_val_bound = [val for i,val in enumerate(BTemp_lat) if (val>latmin and val<latmax)] 
    lon_bound = [i for i,val in enumerate(BTemp_lon) if (val>lonmin and val<lonmax)]   
    lon_val_bound = [val for i,val in enumerate(BTemp_lon) if (val>lonmin and val<lonmax)] 
    return[lat_bound[0],lat_bound[-1],lon_bound[0],lon_bound[-1]]

#% Get idices in accordance with brightness temperature images
IMAG_RES = 4 #km
DEG_TO_KM = 111 #ratio
LAT_BOUND = [0,60] #NA Basin
LON_BOUND = [-120,0] #NA Basin
DIM_BOUND = get_BTempimage_bound(LAT_BOUND[0],LAT_BOUND[1],LON_BOUND[0],LON_BOUND[1])#incices from BT images

#% Best track for a particular storm based on its serial
# get TC estimated centers
B_tracks = xr.open_dataset(BTDIR+"\\"+"Year.2013.ibtracs_all.v03r10.nc")

B_TC_serials = B_tracks['storm_sn'].values
B_TC_names = B_tracks['name'].values

TC_serial = '2013204N11340'
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

#%
C_i = 590
BTemp_filename = r"merg_"+ time_to_string_without_min(I_year[C_i],I_month[C_i],I_day[C_i],I_hour[C_i]) + r"_4km-pixel.nc4"
    
if I_minute[C_i] == 0:
    #slice out BT images for the current basin
    C_BTemp = xr.open_dataset(IRDIR+ "\\" + BTemp_filename)['Tb'].values[0][DIM_BOUND[0]:DIM_BOUND[1]+1,DIM_BOUND[2]:DIM_BOUND[3]+1]
    C_lat = xr.open_dataset(IRDIR+ "\\" + BTemp_filename)['lat'].values[DIM_BOUND[0]:DIM_BOUND[1]+1]
    C_lon = xr.open_dataset(IRDIR+ "\\" + BTemp_filename)['lon'].values[DIM_BOUND[2]:DIM_BOUND[3]+1]
    #interpolate NaN values in BT images
    mask = np.isnan(C_BTemp)
    C_BTemp[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), C_BTemp[~mask])
  
elif I_minute[C_i] == 30:
    #slice out BT images for the current basin
    C_BTemp = xr.open_dataset(IRDIR+ "\\" + BTemp_filename)['Tb'].values[1][DIM_BOUND[0]:DIM_BOUND[1]+1,DIM_BOUND[2]:DIM_BOUND[3]+1]
    C_lat = xr.open_dataset(IRDIR+ "\\" + BTemp_filename)['lat'].values[DIM_BOUND[0]:DIM_BOUND[1]+1]
    C_lon = xr.open_dataset(IRDIR+ "\\" + BTemp_filename)['lon'].values[DIM_BOUND[2]:DIM_BOUND[3]+1]
    #interpolate NaN values in BT images
    mask = np.isnan(C_BTemp)
    C_BTemp[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), C_BTemp[~mask])

#%
fig = plt.figure()
lat_max = np.round(np.max(C_lat),1)
lat_min = np.round(np.min(C_lat),1)
lon_max = np.round(np.max(C_lon),1)
lon_min = np.round(np.min(C_lon),1)
filename = TC_serial+ "_" + I_name + "_" + time_to_string_with_min(I_year[C_i], I_month[C_i], I_day[C_i], I_hour[C_i], I_minute[C_i])

#% Plot BT image with 3 labels
im = plt.imshow(C_BTemp, extent = (lon_min, lon_max, lat_min, lat_max),  cmap='Greys',origin='lower')


# Best track center
plt.plot(I_lon[C_i],I_lat[C_i],'or', markersize = 2) 

ax = plt.gca()
ax.set_title(filename)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.plot(I_lon[C_i],I_lat[C_i],'or', markersize = 2)     

#%%
C_binary = np.where(C_BTemp>270,0,C_BTemp)
C_binary = np.where(C_binary>1,1,C_binary)
C_binary8 = C_binary.astype(np.uint8)

C_img = np.where(C_BTemp>270,0,C_BTemp)
C_img = np.where(C_img >1,255,C_binary)

#%
# noise removal
kernel = np.ones((3,3),np.uint8)
C_binary8_m = cv2.morphologyEx(C_binary8,cv2.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv2.dilate(C_binary8_m,kernel,iterations=3)
#C_binary_dt = ndimage.distance_transform_edt(C_binary)
C_binary_dt = cv2.distanceTransform(C_binary8,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(C_binary_dt,0.02*C_binary_dt.max(),255,0)
sure_fg = np.uint8(sure_fg)
ret, markers = cv2.connectedComponents(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
w_markers = watershed(-C_binary_dt,markers, mask = C_binary8)
#%
fig = plt.figure()
lat_max = np.round(np.max(C_lat),1)
lat_min = np.round(np.min(C_lat),1)
lon_max = np.round(np.max(C_lon),1)
lon_min = np.round(np.min(C_lon),1)
filename = TC_serial+ "_" + I_name + "_" + time_to_string_with_min(I_year[C_i], I_month[C_i], I_day[C_i], I_hour[C_i], I_minute[C_i])

plt.subplot(221)
#% Plot BT image with 3 labels
im = plt.imshow(C_binary8, extent = (lon_min, lon_max, lat_min, lat_max),  cmap='Greys_r',origin='lower')
# Best track center
plt.plot(I_lon[C_i],I_lat[C_i],'or', markersize = 2) 
ax = plt.gca()
ax.set_title(filename)
ax.set_xlabel('Longitude')

plt.subplot(222)
#% Plot BT image with 3 labels
im = plt.imshow(C_binary_dt, extent = (lon_min, lon_max, lat_min, lat_max),  cmap='Greys_r',origin='lower')
plt.subplot(223)
im = plt.imshow(sure_fg, extent = (lon_min, lon_max, lat_min, lat_max),  cmap='Greys_r',origin='lower')
plt.subplot(224)
im = plt.imshow(w_markers, extent = (lon_min, lon_max, lat_min, lat_max),origin='lower')
# Best track center
plt.plot(I_lon[C_i],I_lat[C_i],'or', markersize = 2) 
ax = plt.gca()
ax.set_title(filename)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')


ax.set_ylabel('Latitude')
plt.show()    



