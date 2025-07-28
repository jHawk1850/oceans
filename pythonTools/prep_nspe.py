import numpy as np
import pandas as pd
import os, sys
from pathlib import Path
from glob import glob
import argparse
from netCDF4 import Dataset as netcdf
import configparser
from pyproj import Geod
from scipy.signal import find_peaks, peak_prominences, peak_widths
import tools27
import pdb

#def prep_nspe(system):
def prep_nspe(iniPath):
    """
    prep_nspe(system)
    
    Parameters
    ----------
    system : string
        obs system where iniFiles are.

    Returns
    -------
    None.

    """
    config = configparser.ConfigParser()
    inifiles = list(iniPath.glob('*.ini'))
    #pdb.set_trace()
    for f in inifiles:
        config.read(f)
        system = config.get('General','system')
        season = config.get('General','season')
        sensor = config.get('General','locid')
        pePath = config.get('Paths','pefilepath')
        os.chdir(pePath)
        #
        #
        #
        brngdel = 4
        brngs = np.arange(0,362,brngdel)
        freqs = np.array([25,50],dtype='float')
        for freq in freqs:
            for brng in brngs:
                #print(brng)
                write_nspe_infile(config,brng,freq)
        #print(Path.cwd())
    oosPath = config.get('Paths','oospath')
    os.chdir(oosPath)
    print( 'returned to {}'.format(os.getcwd()))

def write_nspe_infile(config,bearing,freq):
    """
    write_nspe_inifile(config,bearing)
    """
    
    #
    # General definitions
    #
    system = config.get('General','system')
    sensor = config.get('General','locid')
        
    #
    # Model setup
    #
    spdial = config.get('Model','spdial')
    earth  = config.get('Model','earth')
    layers = config.get('Model','layers')
    source = 'omni'
    
    #
    # Sensor parameters
    #
    lon = config.getfloat('Sensor','longitude')
    lat = config.getfloat('Sensor','latitude')
    depth = config.getfloat('Sensor','depth')
    
    #
    # Target parameters
    #
    radials = config.get('Target','radials')
    freqs   = config.get('Target','frequencies')
    target_depths = config.get('Target','target_depths')

    #
    # Domain
    #
    jday = config.get('Domain','julianday')
    maxRng = config.getfloat('Domain','maxrange')
    
    ##############################
    #
    # extractions
    #
    ##############################
    ###
    ### extract bathy line and load bathy
    ###
    if os.path.isfile('rawbathy.asc'):
        print('removing existing bathy file')
        os.remove('rawbathy.asc')
    
    geod = Geod(ellps='WGS84')
    elon,elat,az = geod.fwd(lon,lat,bearing,maxRng)    
    
    #pdb.set_trace()
    tools27.dbdbv_line(lat,lon,elat,elon,0.25)
    # convert bathy lat/lons to range
    tools27.bathy2meters('rawbathy.ext')
    # check bathy 
    tools27.check_bathy(lat,lon,bearing)
    # load scrubbed bathy, will need for maxRange
    bathy = np.loadtxt('rawbathy.asc')

    ###
    ### ssp extraction and data
    ###
    ssp_file = os.path.join('c:/Users/500138/Documents/projects/oceans/envData/',
            'svp_upflow.asc')
    ssp_data = np.loadtxt(ssp_file,skiprows=6)
    rows,cols = np.shape(ssp_data)    
    dep = ssp_data[:,0]
    ssp = ssp_data[:,3]
       
    ##############################
    #
    # write nspe infile
    #
    ##############################
    
    ### construct filename
    #pdb.set_trace()
    br_id = np.round(bearing)
    ##
    ##
    ###freq = 25 ## temporary fix: in future pass from prep_nspe
    ##
    fname = (system + '_B'+str('{0:03d}'.format(br_id))+'_F'+
             str('{0:03.0f}'.format(freq))+'.in') 
    
    with open(fname,'w') as fid:
        # write title line
        ttl = mk_ttl(system,sensor,lat,lon,bearing,freq,depth)
        ###pdb.set_trace()
        fid.write(ttl)
        fid.write('\n')
        fid.write('{}\n'.format('nspe'))
        fid.write(' {}\n'.format(spdial))
        fid.write('earth:{}\n'.format(earth))
        fid.write('layers:{}\n'.format(layers))
        fid.write('\n')

        maxRng = np.max(bathy[:,0])
        fid.write('range\n')
        fid.write( ' {}\n'.format(maxRng) )
        fid.write('\n')

        fid.write('source\n')

        fid.write( '{}\n'.format(source) )
        if depth < 0: 
            depth = bathy[0,1]
            fid.write( ' {} {}\n'.format(freq,depth))
        else:
            fid.write( ' {} {}\n'.format(freq,depth))

        fid.write('\n')
    
        # write bathymetry
        fid.write('bathymetry\n')
        for line in bathy:
            fid.write('{:12.2f} {:10.2f}\n'.format(line[0],line[1]))
        fid.write('{:10.2f} {:9.2f}\n'.format(-1,-1))

        # write ssp 
        fid.write('\n')
        fid.write('svp\n')
   
        #
        # write svp
        #
        tmpR = np.arange(0.0,maxRng,maxRng/2)
        for r in tmpR:
            fid.write('{:12.2f}\n'.format(r))
            for i in range(rows):
                fid.write('{:12.2f} {:10.2f}\n'.format(dep[i],ssp[i]))
            fid.write('{0:4d} {0:4d}\n'.format(-1,-1))
        fid.write('{0:4d}\n'.format(-1))

        #
        # write bottom loss
        #
        fid.write('\n')
        fid.write('bottom\n')
        fid.write('GEOACOUSTIC\n')
        fid.write('{}\n'.format(0.0))
        fid.write('{:8.1f} {:8.1f}\n'.format(0.,1600.))
        fid.write('{:8.1f} {:8.1f}\n'.format(500.,1600.))
        fid.write('{:3d}{:3d}\n'.format(-1,-1))
        fid.write('{:8.1f}{:8.1f}\n'.format(0.,1.5))
        fid.write('{:8.1f}{:8.1f}\n'.format(500.,1.5))
        fid.write('{:3d}{:3d}\n'.format(-1,-1))
        fid.write('{:8.3f}{:8.3f}\n'.format(0.,0.05))
        fid.write('{:8.3f}{:8.3f}\n'.format(500.,0.05))
        fid.write('{:3d}{:3d}\n'.format(-1,-1))
        fid.write('{:3d} \n'.format(-1))
        
        # write output block
        fid.write('\n')
        fid.write('output\n')
        fid.write('metric\n')
        fid.write('tl\n')
        fid.write('{:10.2f} \n'.format(121.9))
        fid.write('{:10.2f} \n'.format(152.4))
        fid.write('{:10.2f} \n'.format(182.9))
        fid.write('{:5d} \n'.format(-1))
        fid.write('{}\n'.format('hrfa'))
        fid.write('{:10.2f} \n'.format(-15.0))
        fid.write('{}\n'.format('field'))
        fid.write('{}\n'.format('high'))
        fid.write('{:10.2f} \n'.format(2000.0))
        fid.write('{}\n'.format('end'))
        fid.write('\n')
    
        fid.write('{}\n'.format('end'))

    ###    
    ### end of write to nspe.in
    ###
    
def mk_ttl(parid,locID,lat,lon,bear,freq,sensorDepth):
    """
    create first line of NSPE input file
    
    usage: ttl = makeTitle(parid,locID,lat,lon,bear,freq,sensorDepth)
    """
    ttl = '{} {} ({},{}) B{:g} F{:g} ZS{} N'.format(parid,locID,lat,lon,
        bear,freq,sensorDepth)
    #pdb.set_trace()
    s = '{:>'+str(82-len(ttl))+'}'
    ttl = ttl + s.format('RAMGEO'+'\n')
    return ttl

def main():
    parser = argparse.ArgumentParser(description=
        'Input path to ini files.\n')
    parser.add_argument(
        "obsystem", type=str, default=None,
        help="Enter the path to ini files. ex. ENAM")

    args    = parser.parse_args()
    obsystem =  args.obsystem
    iniPath = Path(os.getcwd(),'case_studies',obsystem,'iniFiles')
    #pdb.set_trace()
    #prep_nspe(obsystem)
    prep_nspe(iniPath)

if __name__ == '__main__':
  main()