#
import numpy as np
import pdb
import os
from subprocess import call
from netCDF4 import Dataset as netcdf

#
def makeTitle(parid,locID,lat,lon,bear,freq,sensorDepth):
    """
    create first line of NSPE input file
    
    usage: ttl = makeTitle(parid,locID,lat,lon,bear,freq,sensorDepth)
    """
    ttl = '{} {} ({},{}) B{:g} F{:g} ZS{} N'.format(parid,locID,lat,lon,
        bear,freq,sensorDepth)
    s = '{:>'+np.str(82-len(ttl))+'}'
    ttl = ttl + s.format('RAMGEO'+'\n')
    return ttl


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
    import pdb
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

def xtrct_ssp(slat,slon,jday):
    """  
    ----------
    usage: xtrct_ssp(slat,slon,jday)
    extract ssp from gdem, saves in ascii file 'ssp.ext'

    Parameters
    ----------
    slat = source latitude (origin lat)
    slon = source longitude (origin lon)
    jday = julian day
    
    """
    from subprocess import call
    
    #
    # set environment variables for gdem
    #
    gdemDir = 'c:/cygwin/home/500138/databases/GDEM/GDEM-V_3.0.2_20120123'
    os.environ['OAML_GDEMV_INDEX']=gdemDir+'/oamlgdem.ndx'
    os.environ['OAML_GDEMV_DATA']=gdemDir+'/oamlgdem.dat'
    os.environ['LANDMASK']=gdemDir+'/landmask.dat'

    gdemBinDir = 'c:/cygwin/home/500138/programming/c'
    cmd = gdemBinDir + '/gdem.x'
    cmdstr = ('{} {} {} {}'.format(cmd,slat,slon,jday))

    status = call(cmdstr,shell=True)
    return status

def ssp_rd(extractFile):
    """  
    ----------
    usage: ssp = xtrct_ssp(extractFile)
    convert ssp extract to columnar data: save in 'ssp.ext'

    Parameters
    ----------
    extractFile: ssp extract output file

    """
    ssp = list()
    fid = open(extractFile,'r')

    for i in range(14): fid.readline()
    data = fid.readlines()

    ### close ext file
    fid.close()

    for line in data:
        cols = line.strip().split()
        if len(cols)>=2: ssp.append(cols)

    ssp = np.array(ssp,dtype='float')

    ### write extract data
    np.savetxt('ssp.dat',ssp,fmt='%8.2f %8.2f')
    return ssp
    print('done')

def wrt_nspeIn(params,bear,freq,peFilePath):
    """
    wrt_nspeIn(bear,freq,peFilePath)
    write nspe input file

    parameters:
    params = parameters contained in params.py
    bear = bearing
    freq = frequency
    peFilePath = file path for pe input files
    """
    import os
    import sys
    import glob
    import pdb

    from netCDF4 import Dataset as netcdf
    import numpy as np
    from geographiclib.geodesic import Geodesic
    from pyproj import Geod

    import tools
    from tools import nm2km
    from tools import dbdbv_line as xtrctBathy
    from tools import bathy2meters
    from tools import check_bathy
    from tools import xtrct_ssp
    from tools import ssp_rd
    
    #
    # run parameters
    #
    parid  = params['parid'] 
    locID  = params['locID']
    jday   = params['jday']
    source = 'omni'
    lat    = params['slat']  
    lon    = params['slon']  
    freqs  = params['freqs'] 
    maxRng = params['maxRng'] ## in meters
    depth  = params['depth']
    spdial = params['spdial']
    earth  = params['earth']
    layers = params['layers']
    peFilePath = params['peFilePath']
    
    ##############################
    #
    # extractions
    #
    ##############################

    # extract bathy line
    
    if os.path.isfile('rawbathy.asc'):
        print('removing existing bathy file')
        os.remove('rawbathy.asc')
    #geod = Geodesic.WGS84
    #elat = geod.Direct(lat,lon,bear,1000.0*nm2km(maxRng))['lat2']
    #elon = geod.Direct(lat,lon,bear,1000.0*nm2km(maxRng))['lon2']
    geod = Geod(ellps='WGS84')
    ##pdb.set_trace()
    elon,elat,az = geod.fwd(lon,lat,bear,maxRng)    

    xtrctBathy(lat,lon,elat,elon,0.25)
    # convert bathy lat/lons to range
    bathy2meters('rawbathy.ext')
    # check bathy 
    check_bathy(lat,lon,bear)
    # load scrubbed bathy, will need for maxRange
    bathy = np.loadtxt('rawbathy.asc')

    # extract ssp at source
    #xtrct_ssp(lat,lon,jday)

    # begin write nspe input file
    #peFilePath = './pefiles/'
    ##pdb.set_trace()
    ### NB: just for this case!!!
    #if not os.path.exists(peFilePath):
    #    os.makedirs(peFilePath)
    ##fname =peFilePath+'/'+parid + '_B'+str('{0:03d}'.format(bear))+'_F'+str('{0:03d}'.format(freq))+'.in' 
    br_id = np.int(np.abs(np.round(bear)))
    #fname ='./'+parid + '_B'+str('{0:03d}'.format(br_id))+'_F'+str('{0:03d}'.format(freq))+'.in' 
    fname ='./'+parid + '_B'+str('{0:03d}'.format(br_id))+'_F'+str('{0:03d}'.format(freq))+'.in' 
    #print 'NSPE input file: ',os.getcwd()+'\\'+fname
    print('NSPE input file: ',peFilePath+'\\'+fname)
    os.chdir(peFilePath)
    ##fname ='temp.in' 
    fid = open(fname,'w')

    # write title line
    ttl = tools.makeTitle(parid,locID,lat,lon,bear,freq,depth)
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

    #load ssp
    #ssp = ssp_rd('ssp.ext')
    fid.write('\n')
    fid.write('svp\n')
    
    #
    # write svp
    #
    geod = Geodesic.WGS84

    #tmpR = np.arange(0.0,maxRng,maxRng/3.0)
    tmpR = np.arange(0.0,maxRng,maxRng/2.0)
    #fid = open('temp.ext','w')

    for r in tmpR:
        fid.write('{:12.2f}\n'.format(r))
        elon = geod.Direct(lat,lon,bear,r)['lon2']
        elat = geod.Direct(lat,lon,bear,r)['lat2']
        tools.xtrct_ssp(elat,elon,jday)
        data = np.loadtxt('ssp.ext')
        ###NB: changed for single case
        ###NB: changed for single case
        #ssp = tools.ssp_rd('ssp.ext')
        ###if os.path.isfile('ssp.ext'): os.remove('ssp.ext')
        for line in data:
            #####3333
            ###NB: changed for single case
            ###fid.write('{:12.2f} {:10.2f}\n'.format(line[0],line[3]))
            fid.write('{:12.2f} {:10.2f}\n'.format(line[0],line[1]))
        fid.write('{0:4d} {0:4d}\n'.format(-1,-1))
    fid.write('{0:4d}\n'.format(-1))
    if os.path.isfile('ssp.dat'): os.remove('ssp.dat')

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

    fid.close()

def wrt_latLonArr(slat,slon,bearing,maxnmi):
    """
    usage: wrt_latLonArr(slat,slon,bearing,maxnmi)
    returns lat/lon array 
    """
    from geographiclib.geodesic import Geodesic

    ibear = len(bearing)
    nmi   = np.linspace(0.0,maxnmi,1500)
    LATS = np.zeros((ibear,len(nmi)),dtype='float')
    LONS = np.zeros((ibear,len(nmi)),dtype='float')
    
    geod = Geodesic.WGS84
    for i in range(len(bearing)):
        line = geod.Line(float(slat),float(slon),bearing[i])
        for j in range(len(nmi)):
            LONS[i,j] = line.Position(nm2km(nmi[j])*1000.)['lon2']
            LATS[i,j] = line.Position(nm2km(nmi[j])*1000.)['lat2']
    return LATS,LONS

def wrt_tlArr(peFilePath,maxnmi,bearing,freq):
    """
    usage: wrt_tlArr(peFilePath,maxnmi,bearing,freq)
    write tl lat/lon array input file
    """
    import os
    import sys
    import glob
    import pdb

    import numpy as np

    from tools import nm2km
    ##############################
    #
    # 
    #
    ##############################
    rng = np.linspace(0.001,maxnmi,1500)
    
    pdb.set_trace()
    peFiles = glob.glob(peFilePath+'*B*_F'+str('{0:03d}'.format(freq) )+'.atl')
    ibear = len(bearing)
    irng = len(rng)
    tl60  = np.ones((ibear,irng),dtype='float')*np.nan
    tl400 = np.ones((ibear,irng),dtype='float')*np.nan

    for i in range(ibear):
        ###pdb.set_trace()
        string,tok = peFiles[i].split('_B')
        string,tok = tok.split('_F')
        indx = np.where(bearing==int(string))

        ###print peFiles[i]
        data = np.loadtxt(peFiles[i])
        tl60[indx,:]  = np.nan
        tl400[indx,:] = np.nan
        ###
        ind = np.where(rng<=np.max(data[:,0]))
        ###pdb.set_trace()
        ###
        tl60[indx,ind]  = np.interp(rng[ind],data[:,0],data[:,1]) 
        tl400[indx,ind] = np.interp(rng[ind],data[:,0],data[:,2]) 
        
    return tl60,tl400

def wrt_ubandArr(ubFilePath,maxnmi,bearing,freq):
    """
    usage: ub_tlArr(peFilePath,maxnmi,bearing,freq)
    write tl lat/lon array input file
    """
    import os
    import sys
    import glob
    import pdb

    import numpy as np

    from tools import nm2km
    ##############################
    #
    # 
    #
    ##############################
    rng = np.linspace(0.001,maxnmi,1500)
    
    peFiles = glob.glob(ubFilePath+'*B*_F'+str('{0:03d}'.format(freq) )+'.ub')
    ###pdb.set_trace()
    ibear = len(bearing)
    irng = len(rng)
    tl60  = np.ones((ibear,irng),dtype='float')*np.nan
    tl400 = np.ones((ibear,irng),dtype='float')*np.nan

    for i in range(ibear):
        ###pdb.set_trace()
        string,tok = peFiles[i].split('_B')
        string,tok = tok.split('_F')
        indx = np.where(bearing==int(string))

        ###print peFiles[i]
        data = np.loadtxt(peFiles[i])
        tl60[indx,:]  = np.nan
        tl400[indx,:] = np.nan
        ###
        ind = np.where(rng<=np.max(data[:,0]))
        ###pdb.set_trace()
        ###
        tl60[indx,ind]  = np.interp(rng[ind],data[:,0],data[:,2]) 
        tl400[indx,ind] = np.interp(rng[ind],data[:,0],data[:,5]) 
        
    return tl60,tl400

def wrt_cdf(params,ncfname):
    """
    usage: wrt_cdf(filename)
    """
    import numpy as np
    from netCDF4 import Dataset as netcdf
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

def tlStats(ncfile):
    '''
    usage: tl1stats, tl2stats = tlStats(ncfile)
    compute tl stats from existing analysis cdf file

    Parameters
    ------------
    ncfile: netcdf file from analysis
    '''
    from netCDF4 import Dataset as netcdf
    import matplotlib.pylab as pylab
    
    np.seterr(invalid='ignore')
    print("adding tl stats to "+ncfile)

    ### read existing cdf file for some variables
    ncid     = netcdf(ncfile,'r')
    nmi      = ncid.variables['range'][:]
    tl1      = ncid.variables['TL1'][:]
    tl2      = ncid.variables['TL2'][:]
    freqs    = ncid.variables['freqs'][:]
    soRanges = ncid.variables['soRanges'][:]
    ncid.close()

    ifreq = len(freqs)
    irng  = len(soRanges)

    #### add soRanges dimension
    tl1stats = np.ones((irng,ifreq),dtype='float')*np.nan
    tl2stats = np.ones((irng,ifreq),dtype='float')*np.nan
    for i in range(irng):
        print('stand off range '+str(soRanges[i]))
        imin = np.max(np.where(nmi<=(soRanges[i]-0.8333)))
        imax = np.max(np.where(nmi<=(soRanges[i]+0.8333)))
        for j in range(ifreq):
            ###pdb.set_trace()
            print("Calculating stats for "+str(freqs[j])+"Hz")
            temptl = tl1[j,:,imin:imax]
            rows,cols = np.shape(temptl)
            temptl = np.reshape(temptl,rows*cols)
            temptl = temptl[~np.isnan(temptl)]
            n, bins, patches = pylab.hist(temptl,100,normed=True,cumulative=True)
            try:
                i10 = np.max(np.where(n<0.1))+1
            except:
                i10 = 0
            tl1stats[i,j] = bins[i10]
            
            ################
            temptl = tl2[j,:,imin:imax]
            rows,cols = np.shape(temptl)
            temptl = np.reshape(temptl,rows*cols)
            temptl = temptl[~np.isnan(temptl)]
            n, bins, patches = pylab.hist(temptl,100,normed=True,cumulative=True)
            try:
                i10 = np.max(np.where(n<0.1))+1
            except:
                i10 = 0
            tl2stats[i,j] = bins[i10]
            ###pdb.set_trace()
        
    print('done')
    return tl1stats, tl2stats

def add2cdf(ncfile,soRanges):
    '''
    usage: add2cdf(ncfile)
    append sonar terms, soRanges dimensions and tl stats to existing analysis cdf file

    Parameters
    ------------
    ncfile: netcdf file from analysis
    '''
    # open ncfile in append mode
    ncid = netcdf(ncfile,'a')
    
    # create dimension for soRanges
    ncid.createDimension('soRanges',len(soRanges))
    
    # create variables 
    data = ncid.createVariable('soRanges',np.dtype('float32'),('soRanges'))
    data[:] = soRanges

    data = ncid.createVariable('noise',np.dtype('float32'),('freqs'))
    data[:] = np.array([80,75,75,70])
    
    data = ncid.createVariable('gain',np.dtype('float32'),('freqs'))
    data[:] = np.array([8,7.5,7.5,0])
    
    data = ncid.createVariable('RD',np.dtype('float32'),('freqs'))
    data[:] = np.array([5,5,5,5,])

    tl1stats, tl2stats = tlStats(ncfile)
    data = ncid.createVariable('tl1stats',np.dtype('float32'),('soRanges','freqs'))
    data[:] = tl1stats
    data = ncid.createVariable('tl2stats',np.dtype('float32'),('soRanges','freqs'))
    data[:] = tl2stats
    # close file
    ncid.close()

def calc_rings(slat,slon,ringRange):
    """"
    rlats, rlons = calc_rings(slat,slon,ringRange):
    
    """
    from geographiclib.geodesic import Geodesic

    geod = Geodesic.WGS84

    slat = float(slat)
    slon = float(slon)

    rlons = list()
    rlats = list()
    radials = np.linspace(0.0,360.0,361)
    for rad in radials:
        rlons.append(geod.Direct(slat,slon,rad,nm2km(ringRange)*1000.0)['lon2'])
        rlats.append(geod.Direct(slat,slon,rad,nm2km(ringRange)*1000.0)['lat2'])

    return rlats,rlons

def calc_rings_km(slat,slon,ringRange):
    """"
    rlats, rlons = calc_rings(slat,slon,ringRange):
    
    """
    from geographiclib.geodesic import Geodesic

    geod = Geodesic.WGS84

    slat = float(slat)
    slon = float(slon)

    rlons = list()
    rlats = list()
    radials = np.linspace(0.0,360.0,361)
    for rad in radials:
        rlons.append(geod.Direct(slat,slon,rad,ringRange*1000.0)['lon2'])
        rlats.append(geod.Direct(slat,slon,rad,ringRange*1000.0)['lat2'])

    return rlats,rlons

def btest(theta,freq,n,d):
    """
    """
    from numpy import pi
    from numpy import sin

    lmbda = 1500./freq
    theta = (pi/180.)*theta
    a = (pi*d/lmbda)*sin(theta)
    return (sin(n*a)/(n*sin(a)))**2

def btheta(theta,steer,freq,n,d):
    """
    beef: b(theta) for n element line array
    """
    from numpy import pi
    from numpy import sin

    lmbda = 1500./freq
    ###pdb.set_trace()
    theta = (pi/180.)*theta
    steer = (pi/180.)*steer
    a = (pi*d/lmbda)*(sin(theta)-sin(steer))
    arrayFactor = (sin(n*a)/(n*sin(a)))**2
    return arrayFactor

def btheta_linear(theta,steer,freq,n,d):
    """
    b(theta) for linear array; 
    theta = angle in degrees
    """
    from numpy import pi
    from numpy import sin

    lmbda = 1500./freq
    ln = n*d
    k = (2*pi/lmbda)

    theta = (pi/180.)*theta
    steer = (pi/180.)*steer
    delay = sin(theta)-sin(steer)
    a = (k*ln/2)*delay
    return (sin(a)/a)**2
    

def arrayGain(steer,freq,n,d):
    """
    array gain for array with n elements
    gain = arrayGain(theta_0,freq,n,d)
    --------- 
    theta = angle in radians
    """
    def afactor(theta,steer,freq,n,d):
        """
        afactor = b(theta) for n element line array with steering
        NB: same function as btheta
        """
        from numpy import pi
        from numpy import sin

        lmbda = 1500./freq
        ###pdb.set_trace()
        ###theta = (pi/180.)*theta
        steer = (pi/180.)*steer
        a = (pi*d/lmbda)*(sin(theta)-sin(steer))
        arrayFactor = (sin(n*a)/(n*sin(a)))**2
        return arrayFactor
   
    from scipy.integrate import quad
    from numpy import pi
    from tools import btheta

    gain, err = quad(afactor,0.,2*pi,(steer,freq,n,d))
    return 10*np.log10(gain/(2*pi))

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

def create_shpfile(slon,slat,nmi,filename):
    # Example of creating a point shapefile

    #from osgeo import ogr, osr
    from osgeo import ogr, osr
    import numpy as np
    from pyproj import Geod

    geod = Geod(ellps='WGS84')

    # Identify the driver, path and name of the new shapefile
    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    outShp = filename
    ##pdb.set_trace()
    outDataSource = shpDriver.CreateDataSource(outShp)

    # Identify the spatial reference for the new shapefile
    spat_ref = osr.SpatialReference()
    spat_ref.ImportFromEPSG(4326)

    # Create the schema for the new shapefile
    outLayer = outDataSource.CreateLayer('ring layer', spat_ref, ogr.wkbLineString)

    # Create a text field with twenty-five spaces called "NAME"
    field_name = ogr.FieldDefn("NAME", ogr.OFTString)
    field_name.SetWidth(25)
    outLayer.CreateField(field_name)

    # Create a line feature 
    aLine = ogr.Geometry(ogr.wkbLineString)
    bearing = np.linspace(0.0,360.0,361)
    slon = -74.
    slat = 34.0
    for bear in bearing:
      elon,elat,az = geod.fwd(slon,slat,bear,nm2km(100.0)*1000)
      aLine.AddPoint(elon,elat)

    featureDefn = outLayer.GetLayerDefn()
    outFeature = ogr.Feature(featureDefn)

    # Add the text "Some line" to the field "NAME"
    outFeature.SetField("NAME","Some line")

    # Add the new line geometry
    outFeature.SetGeometry(aLine)

    # Create the new line feature in the shapefile "line_demo.shp"
    outLayer.CreateFeature(outFeature)

    # Close the shapefile
    aLine.Destroy()
    outFeature = None
    outDataSource = None

    return 'wrote shapefileS'

def nm2km(nmi):
    """
    return nmi*1.851989
    """
    return nmi*1.851989

def prep_map(slon,slat,lllon,lllat,urlon,urlat):
    """ """

    import datetime
    import math
    import os
    os.environ['PROJ_LIB'] = r'C:\Program Files\Anaconda3\
      \pkgs\proj4-5.2.0-h6538335_1006\Library\share'

    # BaseMap example by geophysique.be
    # tutorial 01
     
    from mpl_toolkits.basemap import Basemap
    import numpy as np

    #Let's create a basemap around Belgium
    
    #m = Basemap(epsg=4326, llcrnrlat=lllat,urcrnrlat=urlat,llcrnrlon=lllon,urcrnrlon=urlon,
    #    lat_0=23.7,lon_0=-77,resolution='h')
    m = Basemap(epsg=4326, llcrnrlat=lllat,urcrnrlat=urlat,llcrnrlon=lllon,urcrnrlon=urlon,
        lat_0=20.95,lon_0=-156.45,resolution='l')

    return m

def add_features(m,lllon,lllat,urlon,urlat,res):
    """ """
    import pdb
    m.drawmapboundary(fill_color='none',linewidth=0.5)
    m.fillcontinents(color='brown',lake_color='aqua')
    m.drawcoastlines(linewidth=0.5)

    meridians = np.arange(lllon,urlon+res,res)
    parallels = np.arange(lllat,urlat+res,res)
    m.drawmeridians(meridians,labels=[1,0,1,1],linewidth=0.5)
    m.drawparallels(parallels,labels=[1,0,1,1],linewidth=0.5)
