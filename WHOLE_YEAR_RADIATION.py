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

WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project"
IRDIR = WORKPLACE + r"\IRimages2012"
BTDIR = WORKPLACE + r"\2_IBTrACSfiles"
RADDIR = WORKPLACE+ r"\6_CERESdata"
TCRADDIR = WORKPLACE + r"\3_Figures\2012_radiation"
#%
def sum1(input):
    return sum(map(sum, input)) 
#%%

Rdataset_2 = xr.open_dataset(RADDIR + r"\CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4A_Subset_20120501-20121231.nc")
R_time_2 = Rdataset_2['time'].values

totalNA_sw_2 = 0
totalNA_lw_2 = 0
for R_i in range(0,np.shape(R_time_2)[0]):
    a_sw = Rdataset_2.toa_sw_all_1h[R_i,:,:].values
    a_lw = Rdataset_2.toa_lw_all_1h[R_i,:,:].values
    sum_a_sw = sum1(a_sw)
    sum_a_lw = sum1(a_lw)
    totalNA_sw_2 = totalNA_sw_2 + sum_a_sw*12321
    totalNA_lw_2 = totalNA_lw_2 + sum_a_lw*12321
    print (str(R_i))
    
Rdataset_1 = xr.open_dataset(RADDIR + r"\CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4A_Subset_20120101-20120430.nc")
R_time_1 = Rdataset_1['time'].values

totalNA_sw_1 = 0
totalNA_lw_1 = 0
for R_i in range(0,np.shape(R_time_1)[0]):
    a_sw = Rdataset_1.toa_sw_all_1h[R_i,:,:].values
    a_lw = Rdataset_1.toa_lw_all_1h[R_i,:,:].values
    sum_a_sw = sum1(a_sw)
    sum_a_lw = sum1(a_lw)
    totalNA_sw_1 = totalNA_sw_1 + sum_a_sw*12321
    totalNA_lw_1 = totalNA_lw_1 + sum_a_lw*12321
    print (str(R_i))

totalNA_sw = totalNA_sw_1 + totalNA_sw_2
totalNA_lw = totalNA_lw_1 + totalNA_lw_2
#%%
os.chdir(TCRADDIR)
h5files = glob.glob("*.h5")

totalNA_TC_sw = 0
totalNA_TC_lw = 0
for file in h5files:
    Hfile_rad = h5py.File(TCRADDIR+ "\\" + file,'r+')
    H_mask_sw_dur = sum(Hfile_rad['mask_sw_dur'])
    H_mask_lw_dur = sum(Hfile_rad['mask_lw_dur'])
    totalNA_TC_sw = totalNA_TC_sw + H_mask_sw_dur*16
    totalNA_TC_lw = totalNA_TC_lw + H_mask_lw_dur*16
    Hfile_rad.close()
    print (file + " done")
#%
totalNA_TC_sw_pc = totalNA_TC_sw/(totalNA_sw_1 + totalNA_sw_2)*100
totalNA_TC_lw_pc = totalNA_TC_lw/(totalNA_lw_1 + totalNA_lw_2)*100
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