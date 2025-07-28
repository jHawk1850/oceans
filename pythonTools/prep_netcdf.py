import numpy as np
import pandas as pd

from pathlib import Path
import os 
import argparse

from netCDF4 import Dataset as netcdf
import configparser
from pyproj import Geod
import pdb

def prep_netcdf(inifile):
    """
    """

    ###################################
    #
    # prep obs system descriptors
    #
    ###################################
    config = configparser.ConfigParser()
    config.read(inifile)
    
    ###
    # system, seasons, sensor
    ###
    system     = config.get('General','system')
    season     = config.get('General','season')
    sensor     = config.get('General','locid')
    
    ###
    # paths
    ###
    obsPath    = Path(config.get('Paths','obspath'))
    peFilePath = Path(config.get('Paths','pefilepath'))
    
    ###
    # model setup
    ###
    slat    = config.get('Sensor','latitude')  
    slon    = config.get('Sensor','longitude')

    #
    # create bearing, range, frequecies, target depth vectors 
    #
    radials = config.get('Target','radials')
    radials = np.array(radials.split(','),dtype='float')
    b_init,b_del,b_final = radials ###KLUDGE
    bearing = np.arange(b_init,b_final+b_del,b_del)
    #pdb.set_trace()
    
    freqs   = config.get('Target','frequencies')
    freqs   = np.array(freqs.split(','),dtype='float')
    freqs = [25,50]
    
    target_depths = config.get('Target','target_depths')
    target_depths = np.array(target_depths.split(','),dtype='float')

    maxRng  = config.get('Domain','maxrange')
    rng = np.arange(0.0,float(maxRng)+250,250,dtype='f')


    ###################################
    #
    # open a new netCDF file for writing
    #
    ###################################
    #ncfname = obsPath/ system/ season/ sensor/ inifile.stem
    ### TEMP nc files!!!!!!!!!!!!!!!!!!
    ncfname = Path(inifile.stem)
    ncfname = ncfname.with_suffix('.nc')
    ncid    = netcdf(ncfname,'w') 
    
    # global attributes
    ncid.location =  'latitude, longitude: {}, {}'.format(slat,slon)
    ncid.sensor = sensor
    ncid.systemID = system
    
    # create dimensions.
    ncid.createDimension('bearing',len(bearing))
    ncid.createDimension('range',len(rng))
    ncid.createDimension('freqs',len(freqs))

    ###################################
    # create the variable, first argument is name of variable, second is datatype, 
    # third is a tuple with the names of dimensions.
    ###################################
    data = ncid.createVariable('bearing',np.dtype('float'),('bearing'))
    data[:] = bearing

    data = ncid.createVariable('range',np.dtype('float'),('range'))
    ###m2nmi = 1/1852.0
    data[:] = rng

    data = ncid.createVariable('freqs',np.dtype('float'),('freqs'))
    data[:] = freqs

    ######
    # lat/lon 3D data array
    ######
    lats,lons = wrt_latLonArr(slat,slon,bearing,rng)
    
    data = ncid.createVariable('lats',np.dtype('float'),('bearing','range'))
    data[::] = lats

    data = ncid.createVariable('lons',np.dtype('float'),('bearing','range'))
    data[::] = lons

    ######
    # tl 3D data array
    ######
    for i in range(len(target_depths)):
        tl_label = 'TL{:g}'.format(i+1)
        print(tl_label)
        data = ncid.createVariable(tl_label,np.dtype('float'),
                                 ('freqs','bearing','range'),fill_value=-9999)
        for j in range(len(freqs)):
          print(target_depths[i])
          tl = wrt_tlArr(peFilePath,system,bearing,freqs[j],rng,i+1)
          #
          data[j,::] = tl
    
    ncid.close()
    print('done')
    ########################################

def wrt_latLonArr(slat,slon,bearings,rng):
    """
    usage: wrt_latLonArr(slat,slon,bearing,maxnmi)
    returns lat/lon array 
    """
    geod = Geod(ellps='WGS84')
    
    #pdb.set_trace()
    ibear = len(bearings)
    #rng   = np.linspace(0.0,maxrng,1500)
    irng  = len(rng)
    LATS = np.zeros((ibear,irng),dtype='float')
    LONS = np.zeros((ibear,irng),dtype='float')
    
    for i in range(ibear):
        for j in range(irng):
            LONS[i,j],LATS[i,j],az2 = geod.fwd(slon,slat,bearings[i],rng[j])
    
    return LATS,LONS

def wrt_tlArr(peFilePath,system,bearing,freq,rng,target_depth_ind):
    """
    usage: wrt_tlArr(peFilePath,system,bearing,freq,rng,target_depth_ind)
    write tl lat/lon array input file
    """
    ibear = len(bearing)
    irng = len(rng)
    tl  = np.ones((ibear,irng),dtype='float')*np.nan
    
    for i in range(ibear):
        pefile = Path(peFilePath,(system+'_B{:03g}_F{:03g}_03.asc'
            .format(bearing[i],freq)))
        #print(pefile)
        
        data = np.loadtxt(pefile)
        ind = np.where(rng<=np.max(data[:,0]))
        tl[i,ind]  = np.interp(rng[ind],data[:,0],data[:,target_depth_ind]) 
    return  tl

def main():
    parser = argparse.ArgumentParser(
        description='Input path to ini files.\n')
    parser.add_argument(
        "obsystem", type=str, default=None,
        help="Enter the path to ini files. ex. ENAM")

    args    = parser.parse_args()
    obsystem =  args.obsystem
    iniPath = Path(Path.cwd(),'case_studies',obsystem,'iniFiles')
    inifiles = list(iniPath.glob('*.ini'))
    
    #prep_nspe(obsystem)
    for f in inifiles:
        prep_netcdf(f)

if __name__ == '__main__':
  main()