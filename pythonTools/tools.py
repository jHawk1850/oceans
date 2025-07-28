# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 14:31:48 2021

@author: 500138
"""
#
import numpy as np
import pandas as pd
import os, sys, glob
from pathlib import Path,PurePath
from netCDF4 import Dataset as netcdf
import configparser
from pyproj import Geod
from scipy.signal import find_peaks, peak_prominences, peak_widths
import pdb

def setup_config(csvfile):
    """
    setup_config(csvfilename)
    usage: setup_config(csvfilename)
     
    reads csv file into pandas dataframe, sets up file tree, 
    and writes ini file.
    """
    ###
    # input csv file into dataframe
    ###
    df = pd.read_csv(csvfile,sep=",")
    df.head()
    rows,cols = np.shape(df)
    for i in range(rows):
        print(df.loc[0:i])

    ###
    # setup OOS home directory and observatories directory
    ###
    oosDir = Path.cwd()
    oosDir.mkdir(parents='True',exist_ok='True')
    
    obsDir = Path(oosDir,'case_studies')
    #obsDir.mkdir(parents='True',exist_ok='True')
    
    ###
    # setup observatory system/season/system directory tree
    ###
    #seasons = ['Win','Spr','Sum','Fall']
    #months = [1,4,7,10]
    seasons = {'Win':1,'Spr':4,'Sum':7,'Fall':10}
    #for season,mon in seasons.items():
    for seas,mon in seasons.items():
        for index,row in df.iterrows():
            ##
            ## define dirs
            ##
            sysDir = Path(obsDir,row['system'])
            iniDir = Path(sysDir,'iniFiles')
            #pdb.set_trace()
            #sensorDir = Path(sysDir,seas,row['locid'])
            #peDir = Path(sensorDir,'peFiles')
            ##
            ## make dirs
            ##
            ##sensorDir.mkdir(parents='True',exist_ok='True')
            iniDir.mkdir(parents='True',exist_ok='True')
            #pdb.set_trace()
            #peDir.mkdir(parents='True',exist_ok='True')
            season = {seas:mon}
            write_ini(df,iniDir,season)    
    
    
def write_ini(df,iniDirectory,season):
    """
    write_ini(csvfilename)
    usage: write_ini(csvfilename)
     
    reads csv file and writes ini file.
    """
    config = configparser.ConfigParser()
    oosDir = Path.cwd()
    config.read(Path(oosDir,'./utils/config.ini'))
    
    for key,val in season.items(): 
        seas,mon = key,val

    rows,cols = np.shape(df)
    #pdb.set_trace()
    for i in range(rows):
        
        #config['DEFAULT']= {'rmgeo':'rmgeo'}
        config['General'] = {'system':df['system'][i],
                             'season':seas,
                             'month':mon,
                             'locid':df['locid'][i]}
        config['Sensor'] = {'longitude':df['longitude'][i],
                            'latitude':df['latitude'][i],
                            'depth':df['depth (m)'][i]}
        
        iniFilename = df['system'][i]+'_'+seas+'_'+df['locid'][i]+'.ini'
        iniFilename = Path(iniDirectory,iniFilename)
        with open(iniFilename,'w') as configfid:
            config.write(configfid)
        list_config_contents(iniFilename)

    
def list_config_contents(inifile):
    """
    list_config_contents(iniFile)
    """
    config = configparser.ConfigParser()
    print(inifile)
    config.read(inifile)
    for section in config.sections():
        print(section)
        for options in config.options(section):
            print("  %s: %s"%(options,config.get(section,options)))
    print(' ')

def prep_nspe(system):
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
    oosPath = Path.cwd()
    obsPath = Path(Path.cwd(),'case_studies')
    iniPath = Path(oosPath,'case_studies',system,'iniFiles')
    config = configparser.ConfigParser()
    for f in list(iniPath.glob('*.ini')):
        config.read(f)
        system = config.get('General','system')
        season = config.get('General','season')
        sensor = config.get('General','locid')
        pePath = Path(obsPath,system,season,sensor,'peFiles')
        #pePath.mkdir(parents='True',exist_ok='True')
        try:
            pePath.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("Existing folder")
        else:
            print("Folder was created")
        os.chdir(pePath)
        #
        #
        #
        brngdel = 60
        brngs = np.arange(0,362,brngdel)
        freq = 25
        for brng in brngs:
            write_nspe_infile(config,brng,freq)
        print(Path.cwd())
    os.chdir(oosPath)
    print(Path.cwd())
    
def mk_ttl(parid,locID,lat,lon,bear,freq,sensorDepth):
    """
    create first line of NSPE input file
    
    usage: ttl = makeTitle(parid,locID,lat,lon,bear,freq,sensorDepth)
    """
    ttl = '{} {} ({},{}) B{:g} F{:g} ZS{} N'.format(parid,locID,lat,lon,
        bear,freq,sensorDepth)
    s = '{:>'+np.str(82-len(ttl))+'}'
    ttl = ttl + s.format('RAMGEO'+'\n')
    return ttl

def write_nspe_infile(config,bearing,freq):
    """
    write_nspe_inifile(config,bearing)
    """
    #
    # model run labels
    #
    system = config.get('General','system')
    sensor = config.get('General','locid')
        
    #
    # model setup
    #
    spdial = config.get('Model','spdial')
    earth  = config.get('Model','earth')
    layers = config.get('Model','layers')
    source = 'omni'
    
    #
    # sensor parameters
    #
    lon = config.getfloat('Sensor','longitude')
    lat = config.getfloat('Sensor','latitude')
    depth = config.getfloat('Sensor','depth')
    
    #
    # Target parameters
    #
    #pdb.set_trace()
    ##freqs = np.array(config.get('Target','frequencies').split(','),dtype='float')
    
    #
    # Domain
    #
    #jday = config.get('Domain','julianday')
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
    
    dbdbv_line(lat,lon,elat,elon,0.25)
    # convert bathy lat/lons to range
    bathy2meters('rawbathy.ext')
    # check bathy 
    check_bathy(lat,lon,bearing)
    # load scrubbed bathy, will need for maxRange
    bathy = np.loadtxt('rawbathy.asc')
       
    ##############################
    #
    # write nspe infile
    #
    ##############################
    
    ### construct filename
    br_id = np.int(np.abs(np.round(bearing)))
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

        ###
        ### load svp data
        ###
        ssp_file = Path('c:/Users/500138/Documents/OOS/','case_studies',
                        system,'svp_upflow.asc')
        ssp_data = np.loadtxt(ssp_file,skiprows=6)
        rows,cols = np.shape(ssp_data)    
        dep = ssp_data[:,0]
        ssp = ssp_data[:,3]
        tmpR = np.arange(0.0,maxRng,maxRng/2)

        # write ssp 
        fid.write('\n')
        fid.write('svp\n')
   
        #
        # write svp
        #
   
        for r in tmpR:
            fid.write('{:12.2f}\n'.format(r))
            for i in range(rows):
                fid.write('{:12.2f} {:10.2f}\n'.format(dep[i],ssp[i]))
            fid.write('{0:4d} {0:4d}\n'.format(-1,-1))
        fid.write('{0:4d}\n'.format(-1))

        # write bottom loss
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
    
def wrt_latLonArr(slat,slon,bearings,maxrng):
    """
    usage: wrt_latLonArr(slat,slon,bearing,maxnmi)
    returns lat/lon array 
    """
    geod = Geod(ellps='WGS84')
    
    ibear = len(bearings)
    rng   = np.linspace(0.0,maxrng,1500)
    irng  = len(rng)
    LATS = np.zeros((ibear,irng),dtype='float')
    LONS = np.zeros((ibear,irng),dtype='float')
    
    for i in range(ibear):
        for j in range(irng):
            LONS[i,j],LATS[i,j],az2 = geod.fwd(slon,slat,bearings[i],rng[j])
    
    return LATS,LONS


def wrt_tlArr(peFilePath,maxnmi,bearing,freq,target_depth):
    """
    usage: wrt_tlArr(peFilePath,maxnmi,bearing,freq)
    write tl lat/lon array input file
    """
#    import glob
    import pdb

    import numpy as np

    ##############################
    #
    # 
    #
    ##############################
    rng = np.linspace(0.001,maxnmi,1500)
    
    #pdb.set_trace()
    #peFiles = glob.glob(peFilePath+'*B*_F'+str('{0:03d}'.format(freq) )+'.atl')
    peFiles = glob.glob(peFilePath+'/peFiles/*03.asc')
    ibear = len(bearing)
    irng = len(rng)
    tl  = np.ones((ibear,irng),dtype='float')*np.nan
    
    for i in range(ibear):
        ###pdb.set_trace()
        string,tok = peFiles[i].split('_B')
        string,tok = tok.split('_F')
        indx = np.where(bearing==int(string))

        ###print peFiles[i]
        data = np.loadtxt(peFiles[i])
        tl[indx,:]  = np.nan
        ###
        ind = np.where(rng<=np.max(data[:,0]))
        ###pdb.set_trace()
        ###
        #
        # 
        #
        tl[indx,ind]  = np.interp(rng[ind],data[:,0],data[:,target_depth]) 
        
    return tl

def wrt_cdf(params,ncfname):
    """
    usage: wrt_cdf(filename)
    """
    #
    
    parid  = params['parid'] 
    locId  = params['locID']
    lat    = params['slat']  
    lon    = params['slon']  
    bear   = params['bear'] 
    freqs  = params['freqs'] 
    maxRng = params['maxRng']

    # create range variable
    ##pdb.set_trace()
    rng = np.linspace(0.0,maxRng,1500,dtype='f',endpoint=True)

    # open a new netCDF file for writing.
    ncid = netcdf(ncfname,'w',format='NETCDF3_CLASSIC') 

    # global attributes
    ncid.location =  str(params['slat'])+','+str(params['slon'])
    ncid.sensor = params['sensor']
    ncid.systemID = params['parid']
    
    # create dimensions.
    ncid.createDimension('bearing',len(bear))
    ncid.createDimension('range',len(rng))
    ncid.createDimension('freqs',len(freqs))

    # create the variable, first argument is name of variable, second is datatype, 
    # third is a tuple with the names of dimensions.
    data = ncid.createVariable('bearing',np.dtype('float32'),('bearing'))
    data[:] = bear

    data = ncid.createVariable('range',np.dtype('float32'),('range'))
    ###m2nmi = 1/1852.0
    data[:] = rng

    data = ncid.createVariable('freqs',np.dtype('float32'),('freqs'))
    data[:] = freqs

    ######
    # 3D data array
    ######

    data1 = ncid.createVariable('TL1',np.dtype('float32'),('freqs','bearing','range'))
    data2 = ncid.createVariable('TL2',np.dtype('float32'),('freqs','bearing','range'))
    
    pdb.set_trace()
    for i in range(len(freqs)):
        print('Frequency: ',freqs[i])
        tl1,tl2 = wrt_tlArr(params['peFilePath'],maxRng,bear,freqs[i])
        data1[i,::] = tl1
        data2[i,::] = tl2

    ### create noise, gain, and RD profiles

    # close the file.
    ncid.close()
    print('done')

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

######################################################
#
# the following modules are utilities I've found useful
#
######################################################
def dbdbv_Area(slat,wlon,nlat,elon,space):
    """  
    ----------
    usage: xtrctbathy_Area(slat,wlon,nlat,elon,space) 
    extract bathymetry area from DBDBV, saves in ascii file 'rawbathy.ext

    Parameters
    ----------
    slat = south latitude
    wlon = west longitude
    nlat = north latitude
    elon = east longitude
    space = resolution
    
    """
    from subprocess import call
    dbdbvDir = 'C:/Users/500138/Documents/databases/DBDBV/v7.2/'
    cmd = dbdbvDir+'bin/Windows/dbv7_command.exe'
    
    typ = 'area'
    
    data = dbdbvDir+'data/dbdbv7_level0c.h5,'
    data = data+dbdbvDir+'data/dbdbv7_level1c.h5'
    ##data = data+dbdbvDir+'data/dbdbv7_level1c.h5'
    
    minRes = '10m'
    maxRes = '2.0min'
    intrp  = 'bilinear'
    dpthType = 'nominal'
    landFlag = 'no_land_db'
    dpthSng = 'positive'
    transLat = '0'
    units = 'meters'
    coords = 'G'
    
    fn = 'rawbathy.ext'
    fmt = 'CAZ'
    
    cmdstr = ('{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}={}'.
              format(cmd,typ,data,minRes,maxRes,intrp,dpthType,
                     landFlag,dpthSng,transLat,units,coords,slat,wlon,
                     nlat,elon,space,fmt,fn))
    ###return cmdstr
    call(cmdstr,shell=True)

def rd_bathyxtrct(bathyFile):
    """
    usage: lats,lons,bathy = rd_bathyxtrct(bathyFile)
    """
    ###pdb.set_trace()
    fid = open(bathyFile)
    line = fid.readline()
    cols = line.split()
    wlon = cols[0].split(':')[1]
    elon = cols[1].split(':')[1]
    slat = cols[2].split(':')[1]
    nlat = cols[3].split(':')[1]

    line = fid.readline()
    cols = line.split()
    c = cols[0].split(':')[1]
    r = cols[1].split(':')[1]

    lons = np.linspace(float(wlon),float(elon),int(c))
    lats = np.linspace(float(slat),float(nlat),int(r))

    line = fid.readline()
    bathy = list()
    for line in fid:
        cols = line.strip().split()
        for i in range(len(cols)): bathy.append(cols[i])
    bathy = np.array(bathy,dtype='float')
    bathy = bathy.reshape(int(r),int(c))
    bathy = np.flipud(bathy)

    return lats,lons,bathy

def readFF(fname):
    """
    ----------
    usage: rng,dpt,z,tl = readFF(fname)    
    reads the full field output from NSPE

    Parameters
    ----------
    fname : full field filename
 
    Returns 
    ----------
    rng : range variable of size r
    dpt : bathymetry along rng
    z   : depth variable of size c
    tl  : 2D transmission loss size rxc
        
    """
    from numpy import fromfile as fread
         
    #pdb.set_trace()

    fid = open(fname,'rb')
    fread(fid,dtype='int',count=1)
    fread(fid,dtype='S16',count=1)
    fread(fid,dtype='S64',count=1)
    nz   = fread(fid,dtype=np.int16,count=1)
    nz = nz[0]
    z     = fread(fid,dtype='f',count=nz)
    
    fread(fid,dtype='S9',count=1)
    fread(fid,dtype='f',count=1)
    fread(fid,dtype='f',count=1)
    fread(fid,dtype='f',count=1)
    nb2     = fread(fid,dtype=np.int16,count=1)
    
    br4     = fread(fid,dtype='f',count=2*np.int(nb2))
    br4     = np.reshape(br4,(2,np.int(nb2)))
    
    fread(fid,dtype='S300',count=1)
    dum     = fread(fid,dtype='int',count=1)
    
    rng = []
    dpt = []
    tl  = []
    
    while True:
        dum = fread(fid,dtype='int',count=1);
        if not dum:
            break
        rng.append(fread(fid,dtype='f',count=1))
        dpt.append(fread(fid,dtype='f',count=1))
        tl.append(fread(fid,dtype='f',count=nz))
        dum = fread(fid,dtype='int',count=1);
    
    rng = np.asarray(rng)
    dpt = np.asarray(dpt)
    tl  = np.asarray(tl)
    ###tl  = np.reshape(tl)    
    fid.close()
    return np.squeeze(rng),np.squeeze(dpt),z,tl.transpose()    

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

def nm2km(nmi):
    """
    return nmi*1.851989
    """
    return nmi*1.851989

def tlpks(tl,slat,slon,brngs,maxrng):
    """
    return tl peak lat/lons 
    """
    geod = Geod(ellps='WGS84')
    rng = np.linspace(0.001,maxrng,1500)
    lats,lons = list(),list()
    rows,cols = np.shape(tl)
    for i in range(rows):
        y = -tl[i,:]
        peaks,props = find_peaks(y,height=-85)
        x = np.max(rng[peaks])
        elon,elat,az = geod.fwd(slon,slat,brngs[i],x)
        lats.append(elat),lons.append(elon)
    return lats,lons

def list_dirs(inpath):
    """
    list_dirs, analog to linux tree
    """
    for root, dirs, files in os.walk(inpath):
        level = root.replace(inpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))