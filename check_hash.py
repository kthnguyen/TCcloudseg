# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:47:05 2018

@author: z3439910
"""

xi = 4**2 + 1
yi = 5**2 + 1
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