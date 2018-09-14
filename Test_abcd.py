# -*- coding: utf-8 -*-
"""
Crexted on Fri Sep 14 07:50:06 2018

@xuthor: z3439910
"""
import numpy as np

x = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1718,19,20])
x= x**2+1
b = np.copy(x)

for xi in x:
    for bi in b:
        if xi != bi:
            for ci in x:
                for di in b:
                    if ci !=di:
                        results = (xi**2+bi**2) - ((xi)*(ci) + (bi)*(di))
                        if (xi!=ci) and (bi!=di) and (results == 0):
                            print ("equal zero at " + str(xi) + " "+ str(bi) + " "+ str(ci) + " "+ str(di))