# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 10:11:39 2022

@author: 500138
"""
import numpy as np
import pandas as pd
import os, sys, glob
from netCDF4 import Dataset as netcdf
import configparser
from pyproj import Geod
#from scipy.signal import find_peaks, peak_prominences, peak_widths

def dbdbv_line(slat,slon,elat,elon,space):
    """  
    ----------
    usage: dbdbv_line(slat,slon,elat,elon,space)
    extract bathymetry line from DBDBV, saves in ascii file 'rawbathy.ext'

    Parameters
    ----------
    slat = source latitude (origin lat)
    slon = source longitude (origin lon)
    elat = end latitude
    elon = end longitude
    space = resolution
    
    """
    from subprocess import call
    dbdbvDir = 'C:/Users/500138/Documents/databases/DBDBV/v7.2/'
    cmd = dbdbvDir+'bin/Windows/dbv7_command.exe'
    
    typ = 'line'
    
    data = dbdbvDir+'data/dbdbv7_level0c.h5,'
    data = data+dbdbvDir+'data/dbdbv7_level1c.h5'
    #data = dbdbvDir+'data/dbdbv7_level0c_0.h5,'
    #data = data+dbdbvDir+'data/dbdbv7_level0c_1.h5,'
    #data = data+dbdbvDir+'data/dbdbv7_level1c_0.h5,'
    #data = data+dbdbvDir+'data/dbdbv7_level1c_1.h5,'
    #data = data+dbdbvDir+'data/dbdbv7_level1c_2.h5'
    ###pdb.set_trace()
    
    minRes = '10m'
    maxRes = '2.0min'
    intrp  = 'bilinear'
    dpthType = 'nominal'
    landFlag = 'no_land_db'
    dpthSng = 'positive'
    transLat = '0'
    units = 'meters'
    coords = 'G'
    track = 'geodesic'
    
    fn = 'rawbathy.ext'
    fmt = 'YXZ'
    
    cmdstr = ('{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}={}'.format(cmd,typ,data,minRes,maxRes,intrp,dpthType,
        landFlag,dpthSng,transLat,units,coords,slat,slon,elat,elon,space,track,fmt,fn))
    #print cmdstr
    call(cmdstr,shell=True)

#
def bathy2meters(extfname):
    """
    ----------
    usage: bathy2meters(extfname)
    converts lat/lon v. parameter to rng(m) v. parameter     

    Parameters
    ----------
    extfname : extract file name (ex. rawbathy.ext)

    """
    ####
    ####from geographiclib.geodesic import Geodesic     
    import pdb
    import pyproj
    
    geod = pyproj.Geod(ellps='WGS84') 
    data = np.loadtxt(extfname)
    r,c = np.shape(data)
    rng = list()
    for i in range(r):
        #rng.append(geod.Inverse(data[0,0],data[0,1],data[i,0],data[i,1])['s12'])
        az1,az2,dist = geod.inv(data[0,1],data[0,0],data[i,1],data[i,0])
        rng.append(dist)
    
    ###pdb.set_trace()    
    bathy = np.zeros((r,2),dtype='f')
    bathy[:,0] = rng
    bathy[:,1] = data[:,2]
    np.savetxt('rawbathy.asc',bathy,fmt='%12.4f')

def check_bathy(slat,slon,bearing):
    """

    """
    ##########
    # load range,depth data and check for land and shallow water
    ##########
    
    data = np.loadtxt('rawbathy.asc')
    #ind = np.where(data[:,1]==-10)
    #if np.size(ind)>0:
    #    ind = np.max(ind)
    #    data = data[0:ind,0:ind]
    ind = np.where(data[:,1]<10)
    if np.size(ind)>0:
        ind = np.min(ind)
        data = data[0:ind,0:ind]

    ### interpolate to regularly space ranges
    
    u,ind  = np.unique(data[:,1],return_index=True)
    ind    = np.sort(ind)
    unq_r  = data[ind,0]
    unq_z  = data[ind,1]
    maxrng = np.max(data[:,0])
    ##rng    = np.linspace(0.0,maxrng,1515)
    rng    = np.linspace(0.0,maxrng,512)
    z      = np.interp(rng,unq_r,unq_z)

    fid = open('rawbathy.asc','w')
    ilen = np.size(rng)
    for i in range(ilen):
        fid.write('{:12.2f} {:10.4f}\n'.format(rng[i],z[i]))
    fid.close()

    os.remove('rawbathy.ext')
    os.remove('rawbathy.ext_SECURITY_README.txt')
    
    print("scrubbed bathy")

def ncdump(nc_fid, verb=False):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print('\t\t%s:' % ncattr,\
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    else:
        print("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print("NetCDF dimension information:")
        for dim in nc_dims:
            print("\tName:", dim) 
            print("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    else:
        print("NetCDF dimension information:")
        for dim in nc_dims:
            print("\tName:", dim) 
            print("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print("NetCDF variable information:")
        for var in nc_vars:
            #if var not in nc_dims:
                print('\tName:', var)
                print("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    else:
        print("NetCDF variable information:")
        for var in nc_vars:
            #if var not in nc_dims:
                print('\tName(dimension):', var, nc_fid.variables[var].dimensions)
                print_ncattr(var)

    return ###nc_attrs, nc_dims###, nc_vars
