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


TC_serial = '2012215N12313'
I_i_max = 419
IMAG_RES = 4 #km
DEG_TO_KM = 111 #ratio
LAT_BOUND = [-20,60] #NA Basin
LON_BOUND = [-120,0] #NA Basin
r = 500


#    TB_thres = 260

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
#I_name = "UNNAMED"
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
TB_thres_list = list(range(230,285,5))

mask_sw = []
mask_lw = []
mask_sw_pc_dur = []
mask_lw_pc_dur = []
for TB_thres in TB_thres_list:
    SAVDIR_RAD = WORKPLACE + r"\3_Figures\\" + TC_serial + "_" + I_name + r"\Radiation" + str(TB_thres)
    HFILE_RAD_DIR = SAVDIR_RAD + r"\\" + TC_serial + r"_" + I_name + r'_radiation'+ str(TB_thres)+'.h5'
    Hfile_rad = h5py.File(HFILE_RAD_DIR,'r+')
    
    H_sum_sw_dur = Hfile_rad['sum_sw_dur']
    H_sum_sw_dur_filled = np.where(np.isnan(H_sum_sw_dur),0,H_sum_sw_dur)
    H_sum_sw_dur[:] = H_sum_sw_dur_filled
    
    H_sum_lw_dur = Hfile_rad['sum_lw_dur']
    H_sum_lw_dur_filled = np.where(np.isnan(H_sum_lw_dur),0,H_sum_lw_dur)
    H_sum_lw_dur[:] = H_sum_lw_dur_filled
    
    
    H_mask_sw_dur = Hfile_rad['mask_sw_dur']
    H_mask_sw_dur_filled = np.where(np.isnan(H_mask_sw_dur),0,H_mask_sw_dur)
    H_mask_sw_dur[:] = H_mask_sw_dur_filled
    mask_sw.append(sum(H_mask_sw_dur))
    
    H_mask_lw_dur = Hfile_rad['mask_lw_dur']
    H_mask_lw_dur_filled = np.where(np.isnan(H_mask_lw_dur),0,H_mask_lw_dur)
    H_mask_lw_dur[:] = H_mask_lw_dur_filled 
    mask_lw.append(sum(H_mask_lw_dur))
    
    H_mask_pc_sw = Hfile_rad['mask_pc_sw']
#    mask_sw_pc.append(sum(H_mask_pc_sw))
    H_mask_pc_sw_filled = np.where(np.isnan(H_mask_pc_sw),0,H_mask_pc_sw)
    H_mask_pc_sw[:] = H_mask_pc_sw_filled
#    mask_sw_pc = sum(H_mask_pc_sw)
    
    H_mask_pc_lw = Hfile_rad['mask_pc_lw']
    H_mask_pc_lw_filled = np.where(np.isnan(H_mask_pc_lw),0,H_mask_pc_lw)
    H_mask_pc_lw[:] = H_mask_pc_lw_filled
#    mask_lw_pc.append(sum(H_mask_pc_lw))
    
    mask_sw_pc_dur.append((sum(H_mask_sw_dur)*16)/(sum(H_sum_sw_dur)*12321))
    mask_lw_pc_dur.append((sum(H_mask_lw_dur)*16)/(sum(H_sum_lw_dur)*12321))
    Hfile_rad.close()
#%%
filename = TC_serial+ "_" + I_name + "_compared"
fig = plt.figure()
plt.plot(TB_thres_list,[i*100 for i in mask_sw_pc_dur])
ax = plt.gca()
ax.set_title(filename + "_Shortwave Contribution NA Ocean")
ax.set_xlabel("Brightness Temperture Threshold (K)") 
ax.set_ylabel('Contribution (Percent)')             
fig.savefig(SAVDIR_RAD + "\\" + filename+ "_SW.png",dpi=500)
#plt.show()
#%
#%%
filename = TC_serial+ "_" + I_name + "_compared"
fig = plt.figure()
plt.plot(TB_thres_list,[i*100 for i in mask_lw_pc_dur])
ax = plt.gca()
ax.set_title(filename + "_Longwave Contribution NA Ocean")
ax.set_xlabel("Brightness Temperture Threshold (K)") 
ax.set_ylabel('Contribution (Percent)')                
fig.savefig(SAVDIR_RAD + "\\" + filename+ "_LW.png",dpi=500)
#plt.show()
#%%
filename = TC_serial+ "_" + I_name + "_compared"
fig = plt.figure()
plt.plot(TB_thres_list,[i*16/(10**9) for i in mask_sw])
ax = plt.gca()
ax.set_title(filename + "_Shortwave Radiation NA Ocean")
ax.set_xlabel("Brightness Temperture Threshold (K)") 
ax.set_ylabel('Total Radiation (GW)')             
fig.savefig(SAVDIR_RAD + "\\" + filename+ "_SWrad.png",dpi=500)
#plt.show()
#%%
filename = TC_serial+ "_" + I_name + "_compared"
fig = plt.figure()
plt.plot(TB_thres_list,[i*16/(10**9) for i in mask_lw])
ax = plt.gca()
ax.set_title(filename + "_Longwave Radiation NA Ocean")
ax.set_xlabel("Brightness Temperture Threshold (K)") 
ax.set_ylabel('Total Radiation (GW)')            
fig.savefig(SAVDIR_RAD + "\\" + filename+ "_LWrad.png",dpi=500)
#plt.show()
#%%
for TB_thres in TB_thres_list:
    print (str(TB_thres))