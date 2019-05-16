# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 21:56:24 2018

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
import glob, os
import pickle


IMAG_RES = 4 #km
DEG_TO_KM = 111 #ratio
r = 500
BASIN = "SI"
CHOSEN_YEAR = "2012"

WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project"
IRDIR = WORKPLACE + r"\IRimages2012"
BTDIR = WORKPLACE + r"\2_IBTrACSfiles"
RADDIR = r"K:\CERES_data\2012"
TCRADDIR = r"K:\THEJUDGEMENT\BASINS_RESULTS_2012" + "\\"+ BASIN + r"\RADIATION"


    #% Set basin boundaries
if BASIN == "WP":
    LAT_BOUND = [-20,60] #WP Basin
    LON_BOUND = [60,180] #WP Basin
    B_tracks = xr.open_dataset(BTDIR+"\\"+"IBTrACS.WP.v04r00.nc")
elif BASIN == "EP": 
    LAT_BOUND = [-20,60] #WP Basin
    LON_BOUND = [-180,-60] #WP Basin
    B_tracks = xr.open_dataset(BTDIR+"\\"+"IBTrACS.EP.v04r00.nc")
elif BASIN == "NI": 
    LAT_BOUND = [-20,60] #WP Basin
    LON_BOUND = [0,120] #WP Basin
    B_tracks = xr.open_dataset(BTDIR+"\\"+"IBTrACS.NI.v04r00.nc")
elif BASIN == "SI": 
    LAT_BOUND_RAD = [-50,0] #WP Basin
    LON_BOUND_RAD = [30,90] #WP Basin
    B_tracks = xr.open_dataset(BTDIR+"\\"+"IBTrACS.SI.v04r00.nc")

#%
def sum1(input):
    return sum(map(sum, input)) 

#%%

def get_CERESimage_bound(latmin,latmax,lonmin,lonmax):
    Rdataset = xr.open_dataset(RADDIR + r"\CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4A_Subset_20120101-20120316.nc")
    R_lat = Rdataset['lat'].values[:]
    R_lon = Rdataset['lon'].values[:]
    lat_bound = [i for i,val in enumerate(R_lat) if (val>latmin and val<latmax)]   
    lat_val_bound = [val for i,val in enumerate(R_lat) if (val>latmin and val<latmax)] 
    lon_bound = [i for i,val in enumerate(R_lon) if (val>lonmin and val<lonmax)]   
    lon_val_bound = [val for i,val in enumerate(R_lon) if (val>lonmin and val<lonmax)] 
    return[lat_bound[0],lat_bound[-1],lon_bound[0],lon_bound[-1]]

#%%
DIM_BOUND_RAD = get_CERESimage_bound(LAT_BOUND_RAD[0],LAT_BOUND_RAD[1],LON_BOUND_RAD[0],LON_BOUND_RAD[1])

#%
Rdataset_1 = xr.open_dataset(RADDIR + r"\CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4A_Subset_20120101-20120316.nc")
R_time_1 = Rdataset_1['time'].values

totalNA_sw_1 = 0
totalNA_lw_1 = 0
for R_i in range(0,np.shape(R_time_1)[0]):
    a_sw = Rdataset_1.toa_sw_all_1h[R_i,:,:].values[DIM_BOUND_RAD[0]:DIM_BOUND_RAD[1]+1,DIM_BOUND_RAD[2]:DIM_BOUND_RAD[3]+1]
    a_lw = Rdataset_1.toa_lw_all_1h[R_i,:,:].values[DIM_BOUND_RAD[0]:DIM_BOUND_RAD[1]+1,DIM_BOUND_RAD[2]:DIM_BOUND_RAD[3]+1]
    sum_a_sw = sum1(a_sw)
    sum_a_lw = sum1(a_lw)
    totalNA_sw_1 = totalNA_sw_1 + sum_a_sw*12321
    totalNA_lw_1 = totalNA_lw_1 + sum_a_lw*12321
    print (str(R_i))

#%
Rdataset_2 = xr.open_dataset(RADDIR + r"\CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4A_Subset_20120317-20120531.nc")
R_time_2 = Rdataset_2['time'].values

totalNA_sw_2 = 0
totalNA_lw_2 = 0
for R_i in range(0,np.shape(R_time_2)[0]):
    a_sw = Rdataset_2.toa_sw_all_1h[R_i,:,:].values[DIM_BOUND_RAD[0]:DIM_BOUND_RAD[1]+1,DIM_BOUND_RAD[2]:DIM_BOUND_RAD[3]+1]
    a_lw = Rdataset_2.toa_lw_all_1h[R_i,:,:].values[DIM_BOUND_RAD[0]:DIM_BOUND_RAD[1]+1,DIM_BOUND_RAD[2]:DIM_BOUND_RAD[3]+1]
    sum_a_sw = sum1(a_sw)
    sum_a_lw = sum1(a_lw)
    totalNA_sw_2 = totalNA_sw_2 + sum_a_sw*12321
    totalNA_lw_2 = totalNA_lw_2 + sum_a_lw*12321
    print (str(R_i))

#%
Rdataset_3 = xr.open_dataset(RADDIR + r"\CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4A_Subset_20120601-20120815.nc")
R_time_3 = Rdataset_3['time'].values

totalNA_sw_3 = 0
totalNA_lw_3 = 0
for R_i in range(0,np.shape(R_time_3)[0]):
    a_sw = Rdataset_3.toa_sw_all_1h[R_i,:,:].values[DIM_BOUND_RAD[0]:DIM_BOUND_RAD[1]+1,DIM_BOUND_RAD[2]:DIM_BOUND_RAD[3]+1]
    a_lw = Rdataset_3.toa_lw_all_1h[R_i,:,:].values[DIM_BOUND_RAD[0]:DIM_BOUND_RAD[1]+1,DIM_BOUND_RAD[2]:DIM_BOUND_RAD[3]+1]
    sum_a_sw = sum1(a_sw)
    sum_a_lw = sum1(a_lw)
    totalNA_sw_3 = totalNA_sw_3 + sum_a_sw*12321
    totalNA_lw_3 = totalNA_lw_3 + sum_a_lw*12321
    print (str(R_i))

#%
Rdataset_4 = xr.open_dataset(RADDIR + r"\CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4A_Subset_20120816-20121030.nc")
R_time_4 = Rdataset_4['time'].values

totalNA_sw_4 = 0
totalNA_lw_4 = 0
for R_i in range(0,np.shape(R_time_4)[0]):
    a_sw = Rdataset_4.toa_sw_all_1h[R_i,:,:].values[DIM_BOUND_RAD[0]:DIM_BOUND_RAD[1]+1,DIM_BOUND_RAD[2]:DIM_BOUND_RAD[3]+1]
    a_lw = Rdataset_4.toa_lw_all_1h[R_i,:,:].values[DIM_BOUND_RAD[0]:DIM_BOUND_RAD[1]+1,DIM_BOUND_RAD[2]:DIM_BOUND_RAD[3]+1]
    sum_a_sw = sum1(a_sw)
    sum_a_lw = sum1(a_lw)
    totalNA_sw_4 = totalNA_sw_4 + sum_a_sw*12321
    totalNA_lw_4 = totalNA_lw_4 + sum_a_lw*12321
    print (str(R_i))

#%
Rdataset_5 = xr.open_dataset(RADDIR + r"\CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4A_Subset_20121031-20121231.nc")
R_time_5 = Rdataset_5['time'].values

totalNA_sw_5 = 0
totalNA_lw_5 = 0
for R_i in range(0,np.shape(R_time_5)[0]):
    a_sw = Rdataset_5.toa_sw_all_1h[R_i,:,:].values
    a_lw = Rdataset_5.toa_lw_all_1h[R_i,:,:].values
    sum_a_sw = sum1(a_sw)
    sum_a_lw = sum1(a_lw)
    totalNA_sw_5 = totalNA_sw_5 + sum_a_sw*12321
    totalNA_lw_5 = totalNA_lw_5 + sum_a_lw*12321
    print (str(R_i))

totalNA_sw = totalNA_sw_1 + totalNA_sw_2 + totalNA_sw_3 + totalNA_sw_4 + totalNA_sw_5
totalNA_lw = totalNA_lw_1 + totalNA_lw_2 + totalNA_lw_3 + totalNA_lw_4 + totalNA_lw_5
#%%
os.chdir(TCRADDIR)
h5files = glob.glob("*.h5")

totalNA_TC_sw = 0
totalNA_TC_lw = 0
for file in h5files:
    Hfile_rad = h5py.File(TCRADDIR+ "\\" + file,'r')
    H_mask_sw_dur = sum(Hfile_rad['mask_sw_dur'])
    H_mask_lw_dur = sum(Hfile_rad['mask_lw_dur'])
    totalNA_TC_sw = totalNA_TC_sw + H_mask_sw_dur*16
    totalNA_TC_lw = totalNA_TC_lw + H_mask_lw_dur*16
    Hfile_rad.close()
    print (file + " done")
#%
totalNA_TC_sw_pc = totalNA_TC_sw/(totalNA_sw)*100
totalNA_TC_lw_pc = totalNA_TC_lw/(totalNA_lw)*100
#%%
f = open('2012_radiation_analysis.pckl','wb')
pickle.dump(totalNA_TC_sw,f)
pickle.dump(totalNA_TC_lw,f)
pickle.dump(totalNA_sw,f)
pickle.dump(totalNA_lw,f)
pickle.dump(totalNA_TC_sw_pc,f)
pickle.dump(totalNA_TC_lw_pc,f)
#%%
os.chdir(TCRADDIR)
f = open(TCRADDIR + r"\2012_radiation_analysis.pckl",'rb')
totalNA_TC_sw_p = pickle.load(f)
totalNA_TC_lw_p = pickle.load(f)
totalNA_sw_p = pickle.load(f)
totalNA_lw_p = pickle.load(f)
totalNA_TC_sw_pc_p = pickle.load(f)
totalNA_TC_lw_pc_p = pickle.load(f)
#%%
area = 12321*120*80
steps = 5136+3624
#%%
for R_i in range(0,1):
    a_sw = Rdataset_2.toa_sw_all_1h[R_i,:,:].values
    a_lw = Rdataset_2.toa_lw_all_1h[R_i,:,:].values
    sum_a_sw = sum1(a_sw)
    sum_a_lw = sum1(a_lw)
    totalNA_sw_2 = totalNA_sw_2 + sum_a_sw*12321
    totalNA_lw_2 = totalNA_lw_2 + sum_a_lw*12321
    print (str(R_i))