# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 07:50:25 2018

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


TC_serial = '2012296N14283'
I_i_max = 467
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
    
def convert_coords(coord_array, option):
    if option == "to180":
        for i in range(0,coord_array.size):
            if coord_array[i] >180:
                coord_array[i] = coord_array[i]-360
    if option == "to360":
        for i in range(0,coord_array.size):
            if coord_array[i] <0:
                coord_array[i] = coord_array[i]+360
    
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
    BTemp_lon = BTempimage['lon'].values[:]
    lat_bound = [i for i,val in enumerate(BTemp_lat) if (val>latmin and val<latmax)]   
    lat_val_bound = [val for i,val in enumerate(BTemp_lat) if (val>latmin and val<latmax)] 
    lon_bound = [i for i,val in enumerate(BTemp_lon) if (val>lonmin and val<lonmax)]   
    lon_val_bound = [val for i,val in enumerate(BTemp_lon) if (val>lonmin and val<lonmax)] 
    return[lat_bound[0],lat_bound[-1],lon_bound[0],lon_bound[-1]]
def sum1(input):
    return sum(map(sum, input))    
#% Get idices in accordance with brightness temperature images
DIM_BOUND = get_BTempimage_bound(LAT_BOUND[0],LAT_BOUND[1],LON_BOUND[0],LON_BOUND[1])#incices from BT images

#%
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

SAVDIR_RAD = WORKPLACE + r"\3_Figures\\" + TC_serial + "_" + I_name + r"\Radiation"
os.mkdir(SAVDIR_RAD)

SAVDIR = WORKPLACE + r"\3_Figures\\" + TC_serial + "_" + I_name


DIM_LAT = DIM_BOUND[1]-DIM_BOUND[0] + 1
DIM_LON = DIM_BOUND[3]-DIM_BOUND[2] + 1
DIM_TIME = np.shape(I_time_interpolate['time'])[0]

#% Start spreading 
# open the label HDF5 file
HFILE_DIR = SAVDIR + r"\\" + TC_serial + r"_" + I_name + r'_labels.h5'
Hfile_label = h5py.File(HFILE_DIR,'r+')  
C_label_TC = Hfile_label['label_TC']
C_label_BG = Hfile_label['label_BG']
C_label_nonTC = Hfile_label['label_nonTC']

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

#%
Rdataset = xr.open_dataset(RADDIR + r"\CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4A_Subset_20120501-20121231.nc")

#%
R_time = Rdataset['time'].values
R_year = pd.to_datetime(R_time).year
R_month = pd.to_datetime(R_time).month
R_day = pd.to_datetime(R_time).day
R_hour = pd.to_datetime(R_time).hour
R_minute = pd.to_datetime(R_time).minute

#%
I_i_min = 1
no1_year = I_year[I_i_min]
no1_month = I_month[I_i_min]
no1_day = I_day[I_i_min]
no1_hour = I_hour[I_i_min]
no1_minute = I_minute[I_i_min]

#% Find No 1 R_i
R_i_month = [idx for idx,x in enumerate(R_month) if x == no1_month]
R_i_day = [idx for idx,x in enumerate(R_day) if x == no1_day and (R_i_month.__contains__(idx) == True)]
R_i_hour = [idx for idx,x in enumerate(R_hour) if x == no1_hour and (R_i_day.__contains__(idx) == True)]
R_i_min = R_i_hour[0]

#%
#% FULL ANALYSIS
#I_i_max = 191
rad_matrix_size = np.uint32((I_i_max - I_i_min)/2 + 1)
mask_pc_sw = np.zeros(rad_matrix_size)
mask_pc_lw = np.zeros(rad_matrix_size)
mask_sw_dur = np.zeros(rad_matrix_size)
mask_lw_dur = np.zeros(rad_matrix_size) 
sum_sw_dur = np.zeros(rad_matrix_size) 
sum_lw_dur = np.zeros(rad_matrix_size) 

BTempimage = xr.open_dataset(WORKPLACE+ "\IRimages2012\merg_2012080100_4km-pixel.nc4")
c_lat = BTempimage['lat'].values[DIM_BOUND[0]:DIM_BOUND[1]+1]
c_lon = BTempimage['lon'].values[DIM_BOUND[2]:DIM_BOUND[3]+1]

rad_matrix_idx = 0
for I_i in range(I_i_min,I_i_max+2,2):
#for I_i in range(I_i_min,3,2):
    R_i = R_i_min + (I_i - I_i_min)
    
    c_flag = C_label_TC[I_i,:,:][:]
    convert_coords(c_lon, "to360")
    #%
    a_sw = Rdataset.toa_sw_all_1h[R_i,:,:].values
    a_lw = Rdataset.toa_lw_all_1h[R_i,:,:].values
    a_lat = Rdataset['lat'].values
    a_lon = Rdataset['lon'].values
    
    #%
    c_flag_pos = np.where(c_flag>0)
    c_flag_pos_lat = c_lat[c_flag_pos[0]]
    c_flag_pos_lon = c_lon[c_flag_pos[1]]
    
    #%
    mask_sw = 0
    mask_lw = 0
    for CA_i in range(0, len(c_flag_pos_lat)-1):
        sel_lat = c_flag_pos_lat[CA_i]
        sel_lon = c_flag_pos_lon[CA_i]
        # try catch to avoid edges
        try:
            sel_lat_idx = max(np.where((a_lat<sel_lat))[0])
        except:
            sel_lat_idx = min(np.where((a_lat>sel_lat))[0])
        
        try:    
            sel_lon_idx = max(np.where((a_lon<sel_lon))[0])
        except:
            sel_lon_idx = min(np.where((a_lon>sel_lon))[0])
        sel_sw = a_sw[sel_lat_idx,sel_lon_idx]
        sel_lw = a_lw[sel_lat_idx,sel_lon_idx]
        mask_sw += sel_sw 
        mask_lw += sel_lw
    
    sum_sw = sum1(a_sw)
    sum_lw = sum1(a_lw)
    
    sum_sw_dur[rad_matrix_idx] = sum_sw
    sum_lw_dur[rad_matrix_idx] = sum_lw
    mask_sw_dur[rad_matrix_idx] = mask_sw
    mask_lw_dur[rad_matrix_idx] = mask_lw
    mask_pc_sw[rad_matrix_idx] = (mask_sw*16)/(sum_sw*12321)
    mask_pc_lw[rad_matrix_idx] = (mask_lw*16)/(sum_lw*12321)
    rad_matrix_idx = rad_matrix_idx + 1
    print(str(I_time_interpolate['time'].values[I_i]) + " done")

#%
#% Overall

    
mask_pc_sw_dur = (sum(mask_sw_dur)*16)/(sum(sum_sw_dur)*12321)
#print ("Percentage of shortwave contribution in the NA Ocean during TC life: " + str(mask_pc_sw_dur*100)+ " percent")
mask_pc_lw_dur = (sum(mask_lw_dur)*16)/(sum(sum_lw_dur)*12321)
#print ("Percentage of longwave contribution in the NA Ocean during TC life: " + str(mask_pc_lw_dur*100)+ " percent")
#%


filename = TC_serial+ "_" + I_name
fig = plt.figure()
plt.plot(mask_pc_sw*100)
ax = plt.gca()
ax.set_title(filename + "_Shortwave Contribution in the NA Ocean"+"\n Total: " +str(np.round(mask_pc_sw_dur*100,3)) + "%")
ax.set_xlabel("from "+ str(I_time_interpolate['time'].values[I_i_min]) + " to " + str(I_time_interpolate['time'].values[I_i_max])) 
ax.set_ylabel('Percent')             
fig.savefig(SAVDIR_RAD + "\\" + filename+ "_SW.png",dpi=300)

fig = plt.figure()
plt.plot(mask_pc_lw*100)
ax = plt.gca()
ax.set_title(filename + "_Longwave Contribution in the NA Ocean"+"\n Total: " +str(np.round(mask_pc_lw_dur*100,3)) + "%")
ax.set_xlabel("from "+ str(I_time_interpolate['time'].values[I_i_min]) + " to " + str(I_time_interpolate['time'].values[I_i_max]) )
ax.set_ylabel('Percent')             
fig.savefig(SAVDIR_RAD  + "\\" + filename+ "_LW.png",dpi=300)

#%
#%

HFILE_RAD_DIR = SAVDIR_RAD + r"\\" + TC_serial + r"_" + I_name + r'_radiation.h5'
Hfile_rad = h5py.File(HFILE_RAD_DIR,'w')
Hfile_rad.close()

Hfile_rad = h5py.File(HFILE_RAD_DIR,'r+')
Hfile_rad.create_dataset('sum_sw_dur', shape = (rad_matrix_size,),chunks=True)
Hfile_rad.create_dataset('sum_lw_dur', shape = (rad_matrix_size,),chunks=True)
Hfile_rad.create_dataset('mask_sw_dur', shape = (rad_matrix_size,),chunks=True)
Hfile_rad.create_dataset('mask_lw_dur', shape = (rad_matrix_size,),chunks=True)
Hfile_rad.create_dataset('mask_pc_sw', shape = (rad_matrix_size,),chunks=True)
Hfile_rad.create_dataset('mask_pc_lw', shape = (rad_matrix_size,),chunks=True)
Hfile_rad.close()

Hfile_rad = h5py.File(HFILE_RAD_DIR,'r+')
H_sum_sw_dur = Hfile_rad['sum_sw_dur']
H_sum_sw_dur[:] = sum_sw_dur
H_sum_lw_dur = Hfile_rad['sum_lw_dur']
H_sum_lw_dur[:] = sum_lw_dur

H_mask_sw_dur = Hfile_rad['mask_sw_dur']
H_mask_sw_dur[:] = mask_sw_dur
H_mask_lw_dur = Hfile_rad['mask_lw_dur']
H_mask_lw_dur[:] = mask_lw_dur

H_mask_pc_sw = Hfile_rad['mask_pc_sw']
H_mask_pc_sw[:] = mask_pc_sw
H_mask_pc_lw = Hfile_rad['mask_pc_lw']
H_mask_pc_lw[:] = mask_pc_lw
Hfile_rad.close()

#%%
Hfile_rad = h5py.File(HFILE_RAD_DIR,'r+')
H_sum_sw_dur = Hfile_rad['sum_sw_dur']
a = H_sum_sw_dur[:]
