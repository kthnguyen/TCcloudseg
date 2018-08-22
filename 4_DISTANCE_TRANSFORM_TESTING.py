# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 21:38:08 2018

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
import time
from matplotlib import colors
import numba
from numba import vectorize, float64, int16, guvectorize, jit
import pickle
import h5py
import cProfile

WORKPLACE = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project"
IRDIR = WORKPLACE + r"\IRimages2012"
SAVDIR = WORKPLACE + r"\3_Figures"
BTDIR = WORKPLACE + r"\2_IBTrACSfiles"
os.chdir(IRDIR)
