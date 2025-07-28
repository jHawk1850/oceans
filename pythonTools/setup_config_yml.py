
import numpy as np
import pandas as pd
import os
import shutil
import configparser
import yaml
import argparse

import pdb

def setup_config(csvfile):
    """
    setup_config(csvfilename)
    usage: setup_config(csvfilename)
     
    reads csv file into pandas dataframe, sets up datafile tree, 
    and writes ini file.
    """
    ###
    # csv -> dataframe
    ###
    #pdb.set_trace()
    df = pd.read_csv(csvfile,sep=",")
    df.head()

    ###
    # get observatory path
    ###
    #config = configparser.ConfigParser()
    #iniTemplate = 'c:/Users/500138/Documents/OOS/utils/config.ini'
    #config.read(iniTemplate)
    #obsPath = config.get('Paths','obspath')      
    
    ymlTemplate = 'c:/Users/500138/Documents/OOS/utils/oos_user.yml'
    with open(ymlTemplate,'r') as fid:
        ymldata = yaml.load(fid,Loader=yaml.FullLoader)
        print('done')
    #
    obsPath = ymldata['Paths']['obsPath']

    pdb.set_trace()
    seasons = {'Win':1,'Spr':4,'Sum':7,'Fall':10}
    for seas,mon in seasons.items():
        for i,row in df.iterrows():
            ##
            ## modify config
            ##
            system = df['system'][i]
            season = seas
            sensor = df['locid'][i]
            config['General'] = {'system':system,
                                 'season':season,
                                 'month':mon,
                                 'locid':sensor}
            config['Sensor'] = {'longitude':df['longitude'][i],
                                'latitude':df['latitude'][i],
                                'depth':df['depth (m)'][i]}
            ##
            ## define/create ini dir
            ##
            #iniPath = os.path.join(obsPath,row['system'],'iniFiles')
            iniPath = os.path.join(obsPath,system,'iniFiles')
            if not os.path.exists(iniPath):
                os.makedirs(iniPath)

            ##
            ## define/create additinal paths
            ##
            pePath = os.path.join(obsPath,system,season,sensor,'peFiles')
            if not os.path.isdir(pePath):
                os.makedirs(pePath)
            sonarPath = os.path.join(obsPath,system,season,sensor,'sonar')
            if not os.path.isdir(sonarPath):
                os.makedirs(sonarPath)
            
            #
            # add paths to ini file
            #
            config.set('Paths','peFilepath',pePath)
            config.set('Paths','sonarFilepath',sonarPath)

            #
            # write ini file
            #
            #pdb.set_trace()
            iniFilename = df['system'][i]+'_'+seas+'_'+df['locid'][i]+'.ini'
            iniFilename = os.path.join(iniPath,iniFilename)
            with open(iniFilename,'w') as configfid:
                config.write(configfid)
            list_inifile(iniFilename)

def list_inifile(inifilename):
    """
    list_config_contents(config_object)
    usage: list_config_contents(config_object)
     
    lists ini file.
    """
    config = configparser.ConfigParser()
    config.read(inifilename)
    for section in config.sections():
        print(section)
        for option in config.options(section):
            print(" %s: %s"%(option,config.get(section,option)))
        print(" ")
            
def list_configparser(config):
    for section in config.sections():
        print(section)
        for option in config.options(section):
            print(' {}: {}'.format(option,config.get(section,option)))
    print(" ")

def main():
    parser = argparse.ArgumentParser(description=
        'Input path to csv file.\n')
    parser.add_argument(
        "csvfile", type=str, default=None,
        help="Enter the path to the ice csvfile. ex. opareas/fname.csv")

    args    = parser.parse_args()
    csvfile =  args.csvfile
    csvfile = os.path.join(os.getcwd(),csvfile)
    setup_config(csvfile)

if __name__ == '__main__':
  main()