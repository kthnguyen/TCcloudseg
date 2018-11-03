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
IRDIR = WORKPLACE + r"\IRimages2012"
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
B_tracks = xr.open_dataset(BTDIR+"\\"+"Year.2012.ibtracs_all.v03r10.nc")
B_TC_serials = B_tracks['storm_sn'].values
B_TC_names = B_tracks['name'].values

#TC_serial_list = ["2012147N30284","2012147N30284","2012169N29291","2012176N26272","2012223N14317","2012229N28305","2012234N16315","2012235N11328", "2012242N13333", "2012242N24317"]
#for TC_i in range(0,len(TC_serial_list)):    
#for TC_i in range(0,3): 
TC_serial = "2012285N26288"
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

DIM_LAT = DIM_BOUND[1]-DIM_BOUND[0] + 1
DIM_LON = DIM_BOUND[3]-DIM_BOUND[2] + 1
DIM_TIME = np.shape(I_time_interpolate['time'])[0]

##%% Start spreading 
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
    
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%% WHOLE RUN
    for C_i in range(1,DIM_TIME):
#    for C_i in range(44,45):
        
        #% Acquire BT images
        C_label_TC[C_i,:,:] = np.zeros([DIM_LAT,DIM_LON])
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
    
        #% Apply previous mask to the current frame
#        if C_i == 216:
#            C_flag_prev = C_label_TC[C_i-2,:,:][:] 
#        else:
        C_flag_prev = C_label_TC[C_i-1,:,:][:] 
        C_flag = C_label_TC[C_i,:,:][:]
        C_flag_temp = C_label_TC[C_i,:,:][:]
        
        # calculate how much the BT center is shifted
        I_idx_prev = get_coord_to_idx(I_lat[C_i-1],I_lon[C_i-1])
        I_idx = get_coord_to_idx(I_lat[C_i],I_lon[C_i])
        laty_shift = I_idx[0] - I_idx_prev[0]
        lonx_shift = I_idx[1] - I_idx_prev[1]
        
        
        # shift the previous mask accordingly
        C_flag_prev_idx = np.where(C_flag_prev>0)
        laty_prev_idx = C_flag_prev_idx[0]
        lonx_prev_idx = C_flag_prev_idx[1]
        laty_current_idx = C_flag_prev_idx[0] + laty_shift
        lonx_current_idx = C_flag_prev_idx[1] + lonx_shift
        C_flag_temp[laty_current_idx, lonx_current_idx] = 2
        

#        C_flag_temp [I_idx[0]-r:I_idx[0]+r,I_idx[1]-r:I_idx[1]+r] = C_flag_prev[I_idx_prev[0]-r:I_idx_prev[0]+r,I_idx_prev[1]-r:I_idx_prev[1]+r] 
#        C_flag_temp = C_flag_prev
        # eliminate all value from the previous mask now become > 280
        C_flag_core = np.where(C_BTemp > 280, 0,C_flag_temp)
        C_flag_core = C_flag_core.astype(np.uint8)
        blobs_labels_core = measure.label(C_flag_core,neighbors=4, background=0)
        
        #% Find all separate blobs after applying the previous mask, only keep the max volume blob
#        regions_core = measure.regionprops(blobs_labels_core)
        min_distance_from_centre = 99999999
        C_flag_core_volume = np.count_nonzero(C_flag_core)
        unique_labels = np.unique(blobs_labels_core)
        for label in unique_labels:
#        for label in range(135,136):
            if label >0:
                prop_volume = np.count_nonzero(blobs_labels_core == label)
                if prop_volume > C_flag_core_volume*0.2:
                    props_indices_list = np.argwhere(blobs_labels_core == label) #list of indices
                    prop_lat_y = np.asarray([C_lat[i] for i in props_indices_list[:,0]]) # separate two columns and refer to C_lat and C_lon
                    prop_lon_x = np.asarray([C_lon[i] for i in props_indices_list[:,1]])
                    props_coords_list = np.column_stack((prop_lat_y,prop_lon_x)) # merge back to make 2-D array
                    distances = cdist(np.array([[I_lat[C_i], I_lon[C_i]]]),props_coords_list)
                    props_min_distance_from_centre = min(np.squeeze(distances))
                    if props_min_distance_from_centre < min_distance_from_centre:
                        min_distance_from_centre = props_min_distance_from_centre
                        chosen_label = label
                    
        #%
        C_flag_core = np.where(blobs_labels_core == chosen_label, 2, 0)
            
        
        #% Start spreading
        C_binary = np.where(C_BTemp>280,0,C_BTemp)
        C_binary = np.where(C_binary>1,1,C_binary)
        C_binary_cut = np.zeros([DIM_LAT,DIM_LON])
    #    r = 500 #the bounding box side = 2r
    #    C_binary_cut[I_idx[0]-r:I_idx[0]+r,I_idx[1]-r:I_idx[1]+r] = C_binary[I_idx[0]-r:I_idx[0]+r,I_idx[1]-r:I_idx[1]+r] 
        C_binary8 = C_binary.astype(np.uint8)
        
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(C_binary8,cv2.MORPH_OPEN,kernel, iterations = 2)
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,0)
        ret, sure_fg = cv2.threshold(dist_transform,0.04*dist_transform.max(),255,0)
    
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,0)
        labels_ws = watershed(-dist_transform, C_flag_core, mask=C_binary8)
        
        C_binary8_second = np.where(labels_ws>0, C_binary8, 0)
        
        C_flag_overflow = np.where(labels_ws == 0, 0,labels_ws)
        C_flag_overflow = C_flag_overflow.astype(np.uint8)
    #%de')
        
        #% Compare the mask obtained from previous mask and the current overflow mask
        C_flag_compared = C_flag_overflow - C_flag_core
        blobs_labels_compared = measure.label(C_flag_compared,neighbors=4, background=0) # identify separate blobs
        
        volume_core = np.count_nonzero(C_flag_core)
        C_flag = C_flag_core[:]
        unique_labels = np.unique(blobs_labels_compared)
        
        volume_core = np.count_nonzero(C_flag_core)
        if volume_core < 8000:
            volume_ratio = 5
        elif volume_core > 8000 and volume_core < 15000:
            volume_ratio = 3
        elif volume_core > 15000 and volume_core < 30000:
            volume_ratio = 1.5
        elif volume_core > 30000 and volume_core < 90000:
            volume_ratio = 0.3
        elif volume_core > 90000 and volume_core < 120000:
            volume_ratio = 0.2
        elif volume_core > 120000 :
            volume_ratio = 0.05
        # Go through all separate blobs, if volume less than 50 percent of the core mask then select
        for label_i in unique_labels:
            if label_i > 0:
                volume_label = np.count_nonzero(blobs_labels_compared==label_i)   
                if volume_label < volume_core*volume_ratio:
                    C_flag = np.where(blobs_labels_compared==label_i,C_flag_overflow,C_flag)
#        #%% PLOT RESULTS
#        # Prepare masks with NaN values
#        C_mask_TC = np.where(C_flag == 0, np.NaN , C_flag)
##        C_mask_Core = np.where(C_Core == 0, np.NaN , C_Core)
#        C_mask_TC_temp = np.where(C_flag_core == 0, np.NaN , C_flag_temp)
#        C_mask_TC_compared = np.where(C_flag_compared == 0, np.NaN , C_flag_temp)
#        # Plot
#        fig = plt.figure()
#        lat_max = np.round(np.max(C_lat),1)
#        lat_min = np.round(np.min(C_lat),1)
#        lon_max = np.round(np.max(C_lon),1)
#        lon_min = np.round(np.min(C_lon),1)
#        filename = TC_serial+ "_" + I_name + "_" + time_to_string_with_min(I_year[C_i], I_month[C_i], I_day[C_i], I_hour[C_i], I_minute[C_i])
#        
#        
#        plt.subplot(231)
#        #% Plot BT image with 3 labels
#        im = plt.imshow(C_binary8, extent = (lon_min, lon_max, lat_min, lat_max),  cmap='Greys_r',origin='lower')
#        # Best track center
#        plt.plot(I_lon[C_i],I_lat[C_i],'or', markersize = 2) 
#        
#        plt.subplot(232)
#        #% Plot BT image with 3 labels
#        im = plt.imshow(C_BTemp, extent = (lon_min, lon_max, lat_min, lat_max),  cmap='Greys',origin='lower')
#        im2 = plt.imshow(C_mask_TC_compared, extent = (lon_min, lon_max, lat_min, lat_max), cmap=colors.ListedColormap(['yellow']),origin='lower',alpha=0.3)
##        im2 = plt.imshow(C_mask_Core, extent = (lon_min, lon_max, lat_min, lat_max), cmap=colors.ListedColormap(['green']),origin='lower',alpha=0.3)
#        
#        plt.subplot(233)
#        
#        im = plt.imshow(C_BTemp, extent = (lon_min, lon_max, lat_min, lat_max),  cmap='Greys',origin='lower')
#        im2 = plt.imshow(C_mask_TC_temp, extent = (lon_min, lon_max, lat_min, lat_max), cmap=colors.ListedColormap(['yellow']),origin='lower',alpha=0.3)
##        im2 = plt.imshow(C_mask_Core, extent = (lon_min, lon_max, lat_min, lat_max), cmap=colors.ListedColormap(['green']),origin='lower',alpha=0.3)
#    
#    #    im = plt.imshow(blobs_labels, extent = (lon_min, lon_max, lat_min, lat_max),  cmap=plt.cm.nipy_spectral,interpolation='nearest',origin='lower')
#        plt.subplot(234)
#    
#        im = plt.imshow(C_BTemp,   cmap='Greys',origin='lower')
#        im2 = plt.imshow(C_mask_TC,  cmap=colors.ListedColormap(['yellow']),origin='lower',alpha=0.3)
##        im2 = plt.imshow(C_mask_Core, extent = (lon_min, lon_max, lat_min, lat_max), cmap=colors.ListedColormap(['green']),origin='lower',alpha=0.3)
#        # Best track center
#        plt.plot(I_idx[1],I_idx[0],'or', markersize = 2)  
#        
#        plt.subplot(235)
#        im = plt.imshow(blobs_labels_core, extent = (lon_min, lon_max, lat_min, lat_max),  cmap=plt.cm.nipy_spectral,interpolation='nearest',origin='lower')
#        
#        plt.subplot(236)
##        im = plt.imshow(props_coords_list, extent = (lon_min, lon_max, lat_min, lat_max),  cmap=plt.cm.nipy_spectral,interpolation='nearest',origin='lower')
#
#        plt.show() 
    
        #%
        C_label_TC[C_i,:,:] = C_flag    
        #%
        C_flag_BG = np.where(C_BTemp<280,0,C_BTemp)
        C_flag_BG = np.where(C_flag_BG>0,4,C_flag_BG)
        C_flag_nonTC = np.zeros([DIM_LAT,DIM_LON])
        C_flag_nonTC = np.where(C_flag_BG < 4,3,C_flag_nonTC)
        C_flag_nonTC = np.where(C_flag >0,0,C_flag_nonTC)
        #im2 = plt.imshow(a, cmap='Greys',origin='lower',alpha=0.4)
        
        C_label_TC[C_i,:,:] = C_flag #flag=2
        C_label_nonTC[C_i,:,:] = C_flag_nonTC #flag=3
        C_label_BG[C_i,:,:] = C_flag_BG #flag=4
        
        #% Plot image
        #flag_pos = np.where(c_flag==1)
        C_mask_TC = np.where(C_flag == 0, np.NaN , C_flag)
        C_mask_nonTC = np.where(C_flag_nonTC == 0, np.NaN , C_flag_nonTC)
        C_mask_BG = np.where(C_flag_BG == 0, np.NaN , C_flag_BG)
        #mask_pos = np.where(c_mask>0)
        #c_Tb = Cdataset.Tb[C_i,:,:].values
        
        
        #% plot IR image and the center point
        fig = plt.figure()
        lat_max = np.round(np.max(C_lat),1)
        lat_min = np.round(np.min(C_lat),1)
        lon_max = np.round(np.max(C_lon),1)
        lon_min = np.round(np.min(C_lon),1)
        filename = TC_serial+ "_" + I_name + "_" + time_to_string_with_min(I_year[C_i], I_month[C_i], I_day[C_i], I_hour[C_i], I_minute[C_i])
        
        #% Plot BT image with 3 labels
        im = plt.imshow(C_BTemp, extent = (lon_min, lon_max, lat_min, lat_max),  cmap='Greys',origin='lower')
        im2 = plt.imshow(C_mask_TC, extent = (lon_min, lon_max, lat_min, lat_max), cmap=colors.ListedColormap(['yellow']),origin='lower',alpha=0.3)
#        im3 = plt.imshow(C_mask_nonTC, extent = (lon_min, lon_max, lat_min, lat_max), cmap=colors.ListedColormap(['red']),origin='lower',alpha=0.3)
#        im4 = plt.imshow(C_mask_BG, extent = (lon_min, lon_max, lat_min, lat_max), cmap=colors.ListedColormap(['blue']),origin='lower',alpha=0.3)
        
        # Best track center
        plt.plot(I_lon[C_i],I_lat[C_i],'or', markersize = 2) 
            
        ax = plt.gca()
        ax.set_title(filename)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        fig.savefig(SAVDIR + "\\" + filename +".png",dpi=200)
        plt.close()
        
#        print(filename + " done")
    #    plt.show()            
    #%
    elapsed_time_overall = time.time() - start_time_overall
    print ('Cloud extraction for all done in ' +  time.strftime("%H:%M:%S", time.gmtime(elapsed_time_overall)))
    
    #% CLOSE HDF5 FILES
    Hfile_label.close()
