# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 13:02:00 2018

@author: z3439910
"""

import numpy as np 
import xarray as xr
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from decimal import Decimal
import glob,os


#%%
WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project"
IMDIR = WORKPLACE + r"\IRimages2012"
os.chdir(IMDIR)
files = glob.glob("merg_2012*.nc")

# get TC estimated centers
Btracks = xr.open_dataset(WORKPLACE+r"\2_IBTrACSfiles\Year.2012.ibtracs_all.v03r10.nc")

#Extract variables into arrays
Bname = Btracks['basin'].values
Bserial = Btracks['t]
#Byear = pd.to_datetime(Btime).year
#Bmonth = pd.to_datetime(Btime).month
#Bday = pd.to_datetime(Btime).day
#Bhour = pd.to_datetime(Btime).hour
#Blat = Btracks['lat_for_mapping'].values
#Blon = Btracks['lon_for_mapping'].values

