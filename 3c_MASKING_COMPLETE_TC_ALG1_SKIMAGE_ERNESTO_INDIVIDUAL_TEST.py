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
from skimage import measure, filters
from skimage.feature import peak_local_max
from skimage.morphology import reconstruction
from scipy import signal
import scipy
from skimage.future import graph
import networkx as nx

WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project"
IRDIR = WORKPLACE + r"\IRimages2012"
SAVDIR = WORKPLACE + r"\3_Figures\ERNESTO"
BTDIR = WORKPLACE + r"\2_IBTrACSfiles"
os.chdir(IRDIR)
#%% Get idices in accordance with brightness temperature images
IMAG_RES = 4 #km
DEG_TO_KM = 111 #ratio
LAT_BOUND = [0,60] #NA Basin
LON_BOUND = [-120,0] #NA Basin

#%% Functions
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
    
def get_coord_to_idx(lat_y,lon_x):
    idx_x = np.int(np.round((lat_y - LAT_BOUND[0])*111/4))
    idx_y = np.int(np.round((lon_x - LON_BOUND[0])*111/4))
    return [idx_x,idx_y]

def get_idx_to_coord(idx_x,idx_y): 
    lat_y = idx_x*4/111 + LAT_BOUND[0]
    lon_x = idx_y*4/111 + LON_BOUND[0]
    return [lat_y,lon_x]

def make_graph(grid):
    # get unique labels
    vertices = np.unique(grid)
 
    # map unique labels to [1,...,num_labels]
    reverse_dict = dict(zip(vertices,np.arange(len(vertices))))
    grid = np.array([reverse_dict[x] for x in grid.flat]).reshape(grid.shape)
   
    # create edges
    down = np.c_[grid[:-1, :].ravel(), grid[1:, :].ravel()]
    right = np.c_[grid[:, :-1].ravel(), grid[:, 1:].ravel()]
    all_edges = np.vstack([right, down])
    all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1], :]
    all_edges = np.sort(all_edges,axis=1)
    num_vertices = len(vertices)
    edge_hash = all_edges[:,0] + num_vertices * all_edges[:, 1]
    # find unique connections
    edges = np.unique(edge_hash)
    # undo hashing
    edges = [[vertices[x%num_vertices],
              vertices[x/num_vertices]] for x in edges] 
 
    return vertices, edges
#%%
DIM_BOUND = get_BTempimage_bound(LAT_BOUND[0],LAT_BOUND[1],LON_BOUND[0],LON_BOUND[1])#incices from BT images

#%% Best track for a particular storm based on its serial
# get TC estimated centers
B_tracks = xr.open_dataset(BTDIR+"\\"+"Year.2012.ibtracs_all.v03r10.nc")

B_TC_serials = B_tracks['storm_sn'].values
B_TC_names = B_tracks['name'].values

TC_serial = '2012215N12313'
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

#%% Create an HDF5 file to store label for the current storm
DIM_LAT = DIM_BOUND[1]-DIM_BOUND[0] + 1
DIM_LON = DIM_BOUND[3]-DIM_BOUND[2] + 1
DIM_TIME = np.shape(I_time_interpolate['time'])[0]

Hfile_label = h5py.File(TC_serial + r"_" + I_name + r'_label_displayonly2.h5','w')
Hfile_label.close()

Hfile_label = h5py.File(TC_serial + r"_" + I_name + r'_label_displayonly2.h5','r+')
Hfile_label.create_dataset('label_TC', shape = (DIM_TIME,DIM_LAT,DIM_LON),chunks=True)
Hfile_label.create_dataset('label_nonTC', shape = (DIM_TIME,DIM_LAT,DIM_LON),chunks=True)
Hfile_label.create_dataset('label_BG', shape = (DIM_TIME,DIM_LAT,DIM_LON),chunks=True)

Hfile_label.close()

#%% Start spreading 
# open the label HDF5 file
Hfile_label = h5py.File(TC_serial + r"_" + I_name + r'_label_displayonly2.h5','r+')  
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
C_min_prev_mask_val = np.zeros(DIM_TIME)
#for C_i in range(0,DIM_TIME):
for C_i in range(200,201):
    
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

    #% Initilize core for spreading
    C_flag = C_label_TC[C_i,:,:][:]
    C_flag = np.zeros([DIM_LAT,DIM_LON])
    # first image
    if C_i > 0:    
        
        box_i_w = [i_w for i_w,x_w in enumerate(C_lon) if abs(I_lon[C_i]-x_w) < S_BOUND_DEG]
        box_i_h = [i_h for i_h,x_h in enumerate(C_lat) if abs(I_lat[C_i]-x_h) < S_BOUND_DEG]
        
        #
        for i_w in box_i_w:
            for i_h in box_i_h:
                t_lat = C_lat[i_h]
                t_lon = C_lon[i_w]
                t_btemp = C_BTemp[i_h,i_w]
                if (calcdistance_km(I_lat[C_i], I_lon[C_i], t_lat, t_lon) <S_BOUND_KM) and (np.int(t_btemp)) < 230:
                    C_flag[i_h,i_w] = 1
                    print ('found at ' + str(i_w) + ' and ' + str(i_h))
    # starting from the second image
    else:
        C_flag_prev = C_label_TC[C_i-1,:,:][:]
        idx_prv = np.where(C_flag_prev == 2)
        idx_prv_y = idx_prv[0]
        idx_prv_x = idx_prv[1]
        min_prv_mask_val = 9999
        min_prv_mask_idx = 0
        min_prv_mask_idy = 0
        for i in range(0,np.shape(idx_prv)[1]-1):
            if (min_prv_mask_val > C_BTemp[idx_prv_y[i],idx_prv_x[i]]) and (calcdistance_km(I_lat[C_i], I_lon[C_i], C_lat[idx_prv_y[i]], C_lon[idx_prv_x[i]])<200):
                min_prv_mask_val  = C_BTemp[idx_prv_y[i],idx_prv_x[i]]
                min_prv_mask_idx = idx_prv_x[i]
                min_prv_mask_idy = idx_prv_y[i]
    
        C_min_prev_mask_val[C_i] =  min_prv_mask_val
        C_flag[min_prv_mask_idy,min_prv_mask_idx] = 1
        C_flag[min_prv_mask_idy,min_prv_mask_idx+1] = 1
        C_flag[min_prv_mask_idy,min_prv_mask_idx-1] = 1
        C_flag[min_prv_mask_idy+1,min_prv_mask_idx] = 1
        C_flag[min_prv_mask_idy-1,min_prv_mask_idx] = 1
        C_flag[min_prv_mask_idy,min_prv_mask_idx+2] = 1
        C_flag[min_prv_mask_idy,min_prv_mask_idx-2] = 1
        C_flag[min_prv_mask_idy+2,min_prv_mask_idx] = 1
        C_flag[min_prv_mask_idy-2,min_prv_mask_idx] = 1
        print("Previous mask min at value " + str(min_prv_mask_val) + " K")
    C_Core = C_flag[:]    
    
    #%% Start spreading
    #% First round, sepearate the connect blobs relating to the cyclone
    C_binary = np.where(C_BTemp>280,0,C_BTemp)
    C_binary = np.where(C_binary>1,1,C_binary)
    C_binary8 = C_binary.astype(np.uint8)
    
    
    
    I_idx = get_coord_to_idx(I_lat[C_i],I_lon[C_i])
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(C_binary8,cv2.MORPH_OPEN,kernel, iterations = 2)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,0)
    
#    dist_matrix = dist_transform[I_idx[0]-100:I_idx[0]+100,I_idx[1]-100:I_idx[1]+100]
#    dist_max = np.unravel_index(np.argmax(dist_matrix, axis=None), dist_matrix.shape)
#    dist_max_big = [I_idx[0]-100 + dist_max[0],I_idx[1] - 100 + dist_max[1]]
#    markers_one = np.zeros([DIM_LAT,DIM_LON])
#    markers_one[dist_max_big[0],dist_max_big[1]] = 1
   
    
#    dist_transform_mod = np.where(dist_transform<20,0,dist_transform)
    labels_ws = watershed(-dist_transform, C_flag, mask=C_binary8)
    
    C_binary8_second = np.where(labels_ws>0, C_binary8, 0)
 
    #%% Second round - divide the previous mask into blobs
    seed = np.copy(C_binary8_second)
    seed[1:-1, 1:-1] = C_binary8_second.max()
    mask = C_binary8_second
    
    # fill all holes the do distance transform
    C_binary8_filled = reconstruction(seed, mask, method='erosion')
    C_binary8_filled8 = C_binary8_filled .astype(np.uint8)
    dist_transform_second = cv2.distanceTransform(C_binary8_filled8,cv2.DIST_L2,0)  
   
    # Maximum them find maxima
    maximum_fil_result = ndimage.maximum_filter(dist_transform_second, size=1)
    maximum_coordinates = peak_local_max(maximum_fil_result,min_distance = 20, indices = True)
    markers_two = np.zeros([DIM_LAT,DIM_LON])
    for i in range(0,np.shape(maximum_coordinates)[0]-1):
        markers_two[maximum_coordinates[i,0],maximum_coordinates[i,1]] = dist_transform_second[maximum_coordinates[i,0],maximum_coordinates[i,1]]
#    C_flag = np.where(labels_ws==1,2,labels_ws)
    
    labels_ws_second = watershed(-dist_transform_second, markers_two, mask=C_binary8_second,watershed_line=True)
    blobs_labels = measure.label(labels_ws_second,neighbors=4, background=0)
    
    #%% Third round - Decide which blobs to count
    # set a bounding box around BT center, then take the max pixel
    box_from_BTcenter = dist_transform_second[I_idx[0]-100:I_idx[0]+100,I_idx[1]-100:I_idx[1]+100]
    max_idx_from_box = np.unravel_index(np.argmax(box_from_BTcenter, axis=None), box_from_BTcenter.shape) #idx within box
    max_idx  = [I_idx[0]-100 + max_idx_from_box[0],I_idx[1] - 100 + max_idx_from_box[1]] #idx in the image
    
    # flag the whole blob that contains the max pixel
    C_flag = np.where(blobs_labels==blobs_labels[max_idx[0],max_idx[1]],1,0)
    #%% Create a graph of labelled regions
    label_graph = nx.Graph()
    # hash for later unique results in convolution
    blobs_labels_hash = blobs_labels**2+1
    unique_labels = np.unique(blobs_labels_hash)
    for xi in unique_labels:
        if xi >0 :
            for i in range(0,np.shape(maximum_coordinates)[0]-1):
                        if blobs_labels_hash[maximum_coordinates[i,0],maximum_coordinates[i,1]] == xi:
                            label_graph.add_node(xi)
                            label_graph.nodes[xi]['centre_y'] = maximum_coordinates[i,0]
                            label_graph.nodes[xi]['centre_x'] = maximum_coordinates[i,1]
                            label_graph.nodes[xi]['volume'] = np.count_nonzero(blobs_labels_hash==xi)
        for yi in unique_labels:
#    xi = 4**2 + 1
#    yi = 5**2 + 1
            if yi>xi:
                kernel_xy = np.array([xi,0,yi]).reshape(3,1)
                results1 = scipy.signal.convolve2d(blobs_labels_hash,kernel_xy,mode = 'same')
        
                kernel_xy = np.array([xi,0,yi]).reshape(1,3)
                results2 = scipy.signal.convolve2d(blobs_labels_hash,kernel_xy,mode = 'same')
                
                kernel_xy = np.array([yi,0,xi]).reshape(3,1)
                results3 = scipy.signal.convolve2d(blobs_labels_hash,kernel_xy,mode = 'same')
        
                kernel_xy = np.array([yi,0,xi]).reshape(1,3)
                results4 = scipy.signal.convolve2d(blobs_labels_hash,kernel_xy,mode = 'same')
                 
                results1 = np.where(results1 == (xi**2 + yi**2), results1,0)
                results2 = np.where(results2 == (xi**2 + yi**2), results2,0)
                results3 = np.where(results3 == (xi**2 + yi**2), results3,0)
                results4 = np.where(results4 == (xi**2 + yi**2), results4,0)   
                results = results1 + results2 + results3 + results4
                occurences = np.count_nonzero(results)
                
                label_graph.add_edge(xi,yi)
                label_graph[xi][yi]['border'] = occurences
    
    #%%
    ct_label = blobs_labels_hash[max_idx[0],max_idx[1]]  
    for i in unique_labels:
        i
#%%
    fig = plt.figure()
    lat_max = np.round(np.max(C_lat),1)
    lat_min = np.round(np.min(C_lat),1)
    lon_max = np.round(np.max(C_lon),1)
    lon_min = np.round(np.min(C_lon),1)
    filename = TC_serial+ "_" + I_name + "_" + time_to_string_with_min(I_year[C_i], I_month[C_i], I_day[C_i], I_hour[C_i], I_minute[C_i])
    
    plt.subplot(221)
    #% Plot BT image with 3 labels
    C_mask_TC = np.where(C_flag == 0, np.NaN , C_flag)
    im = plt.imshow(C_binary8, extent = (lon_min, lon_max, lat_min, lat_max),  cmap='Greys_r',origin='lower')
    im2 = plt.imshow(C_mask_TC, extent = (lon_min, lon_max, lat_min, lat_max), cmap=colors.ListedColormap(['yellow']),origin='lower',alpha=0.6)
    # Best track center
    plt.plot(I_lon[C_i],I_lat[C_i],'or', markersize = 2) 
    ax = plt.gca()
    ax.set_title(filename)
    ax.set_xlabel('Longitude')
    
    plt.subplot(222)
    #% Plot BT image with 3 labels
    im = plt.imshow(C_binary8_filled, extent = (lon_min, lon_max, lat_min, lat_max),  cmap='Greys_r',origin='lower')
    plt.subplot(223)
    im = plt.imshow(maximum_fil_result,  cmap=plt.cm.nipy_spectral, interpolation='nearest',origin='lower')

    plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    plt.subplot(224)
    im = plt.imshow(blobs_labels,  cmap=plt.cm.nipy_spectral, interpolation='nearest',origin='lower')
#    C_mask_TC = np.where(C_flag == 0, np.NaN , C_flag)
#    C_mask_Core = np.where(C_Core == 0, np.NaN , C_Core)
#    im = plt.imshow(C_binary8, extent = (lon_min, lon_max, lat_min, lat_max),  cmap='Greys_r',origin='lower')
#    im2 = plt.imshow(C_mask_TC, extent = (lon_min, lon_max, lat_min, lat_max), cmap=colors.ListedColormap(['yellow']),origin='lower',alpha=0.6)
#    im2 = plt.imshow(C_mask_Core, extent = (lon_min, lon_max, lat_min, lat_max), cmap=colors.ListedColormap(['green']),origin='lower',alpha=0.3)
    # Best track center
#    plt.plot(I_lon[C_i],I_lat[C_i],'or', markersize = 2) 
    ax = plt.gca()
    ax.set_title(filename)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    
    ax.set_ylabel('Latitude')
    plt.show()    


    #%%
#    C_label_TC[C_i,:,:] = C_flag    
#    #%
#    C_flag_BG = np.where(C_BTemp<280,0,C_BTemp)
#    C_flag_BG = np.where(C_flag_BG>0,4,C_flag_BG)
#    C_flag_nonTC = np.zeros([DIM_LAT,DIM_LON])
#    C_flag_nonTC = np.where(C_flag_BG < 4,3,C_flag_nonTC)
#    C_flag_nonTC = np.where(C_flag >0,0,C_flag_nonTC)
#    #im2 = plt.imshow(a, cmap='Greys',origin='lower',alpha=0.4)
#    
#    C_label_TC[C_i,:,:] = C_flag #flag=2
#    C_label_nonTC[C_i,:,:] = C_flag_nonTC #flag=3
#    C_label_BG[C_i,:,:] = C_flag_BG #flag=4
#    
#    #% Plot image
#    #flag_pos = np.where(c_flag==1)
#    C_mask_TC = np.where(C_flag == 0, np.NaN , C_flag)
#    C_mask_nonTC = np.where(C_flag_nonTC == 0, np.NaN , C_flag_nonTC)
#    C_mask_BG = np.where(C_flag_BG == 0, np.NaN , C_flag_BG)
#    #mask_pos = np.where(c_mask>0)
#    #c_Tb = Cdataset.Tb[C_i,:,:].values
#    
#    
#    #% plot IR image and the center point
#    fig = plt.figure()
#    lat_max = np.round(np.max(C_lat),1)
#    lat_min = np.round(np.min(C_lat),1)
#    lon_max = np.round(np.max(C_lon),1)
#    lon_min = np.round(np.min(C_lon),1)
#    filename = TC_serial+ "_" + I_name + "_" + time_to_string_with_min(I_year[C_i], I_month[C_i], I_day[C_i], I_hour[C_i], I_minute[C_i])
#    
#    #% Plot BT image with 3 labels
#    im = plt.imshow(C_BTemp, extent = (lon_min, lon_max, lat_min, lat_max),  cmap='Greys',origin='lower')
#    im2 = plt.imshow(C_mask_TC, extent = (lon_min, lon_max, lat_min, lat_max), cmap=colors.ListedColormap(['yellow']),origin='lower',alpha=0.3)
#    im3 = plt.imshow(C_mask_nonTC, extent = (lon_min, lon_max, lat_min, lat_max), cmap=colors.ListedColormap(['red']),origin='lower',alpha=0.3)
#    im4 = plt.imshow(C_mask_BG, extent = (lon_min, lon_max, lat_min, lat_max), cmap=colors.ListedColormap(['blue']),origin='lower',alpha=0.3)
#    
#    # Best track center
#    plt.plot(I_lon[C_i],I_lat[C_i],'or', markersize = 2) 
#    
#    # Previous core position
#    if C_i>0:
#        plt.plot(C_lon[min_prv_mask_idx],C_lat[min_prv_mask_idy],'co', markersize = 2)
#        
#    ax = plt.gca()
#    ax.set_title(filename)
#    ax.set_xlabel('Longitude')
#    ax.set_ylabel('Latitude')
#    fig.savefig(SAVDIR + "\\" + filename +".png",dpi=1000)
#    plt.close()
##    plt.show()            
#
#    elapsed_time_overall = time.time() - start_time_overall
#    print ('Cloud extraction for all done in ' +  time.strftime("%H:%M:%S", time.gmtime(elapsed_time_overall)))
#
##% CLOSE HDF5 FILES
#Hfile_label.close()
