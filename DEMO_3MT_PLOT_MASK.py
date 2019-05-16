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
from skimage import measure
from skimage.feature import peak_local_max
from skimage.morphology import reconstruction
from scipy.spatial.distance import cdist

WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project"
IRDIR = WORKPLACE + r"\IRimages2013"
BTDIR = WORKPLACE + r"\2_IBTrACSfiles"
os.chdir(IRDIR)

#% CONSTANT
IMAG_RES = 4 #km
DEG_TO_KM = 111 #ratio
LAT_BOUND = [-20,60] #NA Basin
LON_BOUND = [-120,0] #NA Basin
r = 500
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

def get_coord_to_idx(lat_y,lon_x):
    idx_x = np.int(np.round((lat_y - LAT_BOUND[0])*111.5/4))
    idx_y = np.int(np.round((lon_x - LON_BOUND[0])*111.5/4))
    return [idx_x,idx_y]

def get_idx_to_coord(idx_x,idx_y): 
    lat_y = idx_x*4/111 + LAT_BOUND[0]
    lon_x = idx_y*4/111 + LON_BOUND[0]
    return [lat_y,lon_x]

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

#%% Best track for a particular storm based on its serial
# get TC estimated centers
B_tracks = xr.open_dataset(BTDIR+"\\"+"Year.2013.ibtracs_all.v03r10.nc")
B_TC_serials = B_tracks['storm_sn'].values
B_TC_names = B_tracks['name'].values

#TC_serial_list = ["2012147N30284","2012147N30284","2012169N29291","2012176N26272","2012223N14317","2012229N28305","2012234N16315","2012235N11328", "2012242N13333", "2012242N24317"]
#for TC_i in range(0,len(TC_serial_list)):    
#for TC_i in range(0,3): 
TC_serial = "2013204N11340"
for i,j in enumerate(B_TC_serials):
    if j.decode("utf-8") == TC_serial:
        I_TC_idx = i
## extract variables into arrays
I_name = B_TC_names[I_TC_idx].decode("utf-8")
I_TC_time = B_tracks['source_time'].values[I_TC_idx,:]
I_TC_time = pd.DataFrame(I_TC_time).dropna().values[:,0]
print ("Starting processing TC " + I_name)


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

SAVDIR = WORKPLACE + r"\3_Figures\\" + TC_serial + "_" + I_name
#SAVDIR = WORKPLACE + r"\3_Figures\\" + TC_serial + "_" + I_name
DIM_LAT = DIM_BOUND[1]-DIM_BOUND[0] + 1
DIM_LON = DIM_BOUND[3]-DIM_BOUND[2] + 1
DIM_TIME = np.shape(I_time_interpolate['time'])[0]

##%% Start spreading 
# open the label HDF5 file
HFILE_DIR = SAVDIR + r"\\" + TC_serial + r"_" + I_name + r'_labels.h5'
Hfile_label = h5py.File(HFILE_DIR,'r')  
C_label_TC = Hfile_label['label_TC']

# define some variables
TB_THRES = 280
start_time_overall = time.time()
# define search boundry
S_BOUND_KM = 300 #km
S_BOUND_DEG = S_BOUND_KM/111 #convert km to deg
S_NO_PX = np.round(S_BOUND_KM/IMAG_RES)

S_BOUND_TOT_KM = 1110 #km
S_BOUND_TOT_DEG = S_BOUND_TOT_KM/111 #convert km to deg
S_NO_TOT_PX = np.round(S_BOUND_TOT_KM/IMAG_RES) 
    
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%% WHOLE RUN
    for C_i in range(144,145):
#    for C_i in range(44,45):
        
        #% Acquire BT images
        
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
    
        C_flag = C_label_TC[C_i,:,:][:]
        C_mask_TC = np.where(C_flag == 0, np.NaN , C_flag)
        
        lat_max = np.round(np.max(C_lat),1)
        lat_min = np.round(np.min(C_lat),1)
        lon_max = np.round(np.max(C_lon),1)
        lon_min = np.round(np.min(C_lon),1)
        filename = TC_serial+ "_" + I_name + "_" + time_to_string_with_min(I_year[C_i], I_month[C_i], I_day[C_i], I_hour[C_i], I_minute[C_i])

        #%
        from matplotlib.ticker import FormatStrFormatter
        fig = plt.figure()
        csfont_tick = {'fontname':'Times New Roman','weight' : 'normal', 'size' : 28}
        csfont_ax = {'fontname':'Times New Roman','weight' : 'normal', 'size' : 30}
        im = plt.imshow(C_BTemp, extent = (lon_min, lon_max, lat_min, lat_max),  cmap='Greys',origin='lower')
        im2 = plt.imshow(C_mask_TC, extent = (lon_min, lon_max, lat_min, lat_max), cmap=colors.ListedColormap(['yellow']),origin='lower',alpha=0.3)
        axes = plt.gca()
        xmin = -55
        xmax = -15
        ymin = -5
        ymax = 35
        axes.set_xlim([xmin,xmax])
        axes.set_ylim([ymin,ymax])
        plt.xlabel('Longitudes',**csfont_ax)
        plt.ylabel('Latitudes',**csfont_ax)
        a = plt.gca()
        a.set_xticklabels(a.get_xticks(), **csfont_tick)
        a.set_yticklabels(a.get_yticks(), **csfont_tick)
        plt.xticks(np.arange(xmin, xmax, 10.0))
        plt.yticks(np.arange(ymin, ymax, 10.0))
        fig.set_tight_layout({"pad": .0})
        plt.plot(I_lon[C_i],I_lat[C_i],'or', markersize = 5) 
        a.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        a.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        fig.savefig(SAVDIR + "\\" + "3_c" +".png",dpi=200)
    #    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        plt.close()
#        plt.show() 
    #%%
    elapsed_time_overall = time.time() - start_time_overall
    print ('Cloud extraction for all done in ' +  time.strftime("%H:%M:%S", time.gmtime(elapsed_time_overall)))
    
    #% CLOSE HDF5 FILES
    Hfile_label.close()
