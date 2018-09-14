# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 13:44:43 2018

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
from skimage.measure import compare_ssim as ssim

WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project"
IRDIR = WORKPLACE + r"\IRimages2013"
SAVDIR = WORKPLACE + r"\3_Figures\HAIYAN"
BTDIR = WORKPLACE + r"\2_IBTrACSfiles"
os.chdir(IRDIR)

#%%
#%% Best track for a particular storm based on its serial
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
#%%
# open the label HDF5 file
Hfile_label = h5py.File('2013204N11340_DORIAN_label.h5','r')  
C_label_TC = Hfile_label['label_TC']
C_label_BG = Hfile_label['label_BG']
C_label_nonTC = Hfile_label['label_nonTC']




#%%
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
 
def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)
 
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
 
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray,origin='lower')
	plt.axis("off")
 
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray,origin='lower')
	plt.axis("off")
 
	# show the images
#%%
def get_BTempimage_bound(latmin,latmax,lonmin,lonmax):
    BTempimage = xr.open_dataset(IRDIR+ "\merg_2013092500_4km-pixel.nc4")
    latmin = 0
    latmax = 60
    BTemp_lat = BTempimage['lat'].values[:]
    BTemp_lon = BTempimage['lon'].values
    lat_bound = [i for i,val in enumerate(BTemp_lat) if (val>latmin and val<latmax)]   
    lat_val_bound = [val for i,val in enumerate(BTemp_lat) if (val>latmin and val<latmax)] 
    lon_bound = [i for i,val in enumerate(BTemp_lon) if (val>lonmin and val<lonmax)]   
    lon_val_bound = [val for i,val in enumerate(BTemp_lon) if (val>lonmin and val<lonmax)] 
    return[lat_bound[0],lat_bound[-1],lon_bound[0],lon_bound[-1]]
    
def get_image(C_i):
    LAT_BOUND = [0,60] #NA Basin
    LON_BOUND = [-120,0] #NA Basin
    imag1 = C_label_TC[C_i,:,:][:]
    i_lat = I_lat[C_i]
    i_lon = I_lon[C_i]
    lat_idx = np.int(np.round((i_lat - LAT_BOUND[0])*111/4))
    lon_idx = np.int(np.round((i_lon - LON_BOUND[0])*111/4))
    imag1_crop = imag1[lat_idx-400:lat_idx + 400,lon_idx-400:lon_idx+400]
    return imag1_crop
#%%
LAT_BOUND = [0,60] #NA Basin
LON_BOUND = [-120,0] #NA Basin
imag1 = C_label_TC[C_i,:,:][:]
i_lat = I_lat[C_i]
i_lon = I_lon[C_i]
lat_idx = np.int(np.round((LAT_BOUND[1] - i_lat)*111/4))
lon_idx = np.int(np.round((i_lon - LON_BOUND[0])*111/4))
imag1_crop = imag1[lat_idx-250:lat_idx + 250,lon_idx-250:lon_idx+250]
plt.imshow(imag1, cmap = plt.cm.gray,origin='lower')
#%%    
C_i = 197
print (str(I_time_interpolate['time'][C_i]))
imag1 = get_image(551)
imag2 = get_image(550)
compare_images(imag1,imag2,'1 and 2')

#%%
Hfile_label.close()