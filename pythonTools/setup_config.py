
import numpy as np
import pandas as pd
from pathlib import Path, PurePath
import os
import sys
import shutil
import configparser
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
    df.head(3)
    pdb.set_trace()
    
    ###
    # get observatory path
    ###
    #pdb.set_trace()
    config = configparser.ConfigParser()
    
    #load ini template
    iniTemplate = Path('c:/Users/500138/Documents/projects/oceans/utils/config.ini')
    config.read(iniTemplate)
    obsPath = config.get('Paths','obspath')      

    # record path to csv file
    csvpath = str(Path(csvfile).resolve())
    config.set('Paths','csvpath',csvpath)

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
            ## define/create additinal paths
            ##
            #pePath = os.path.join(obsPath,system,season,sensor,'peFiles')
            #if not os.path.isdir(pePath):
            #    os.makedirs(pePath)
            #sonarPath = os.path.join(obsPath,system,season,sensor,'sonar')
            #if not os.path.isdir(sonarPath):
            #    os.makedirs(sonarPath)

            ##
            ## define/create ini/sonar/pe dirs
            ##
            iniPath = Path(obsPath,system,'iniFiles')
            iniPath.mkdir(parents=True,exist_ok=True)
            sonarPath = Path(obsPath,system,season,sensor,'sonar')
            sonarPath.mkdir(parents=True,exist_ok=True)
            
            #
            # check if there are pe files
            #
            pePath = Path(obsPath,system,season,sensor,'peFiles')
            try:
                pePath.mkdir(parents=True,exist_ok=False)
            except (FileExistsError):
                sys.exit(str(pePath)+' exists')
            
            
            #
            # add paths to ini file
            #
            config.set('Paths','peFilepath',str(pePath))
            config.set('Paths','sonarFilepath',str(sonarPath))

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
        print('')
    print(' ')

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