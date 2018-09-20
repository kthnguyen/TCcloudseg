# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 00:29:34 2018

@author: z3439910
"""

from PIL import Image
import numpy as np

WORKPLACE2 = r"C:\Users\z3439910\Documents\Kien\1_Projects\2_Msc\1_E1\5_GIS_project\Kaggle_data"
im_frame = Image.open(WORKPLACE2 + r"\img1.png")

im = plt.imshow(im_frame)
plt.show()
