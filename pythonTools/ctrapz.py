# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 10:03:24 2022

@author: 500138
"""
def trap(y,z):
    dx = z[1]-z[0]
    t = (dx/2)*(y[1]+y[0])
    return t