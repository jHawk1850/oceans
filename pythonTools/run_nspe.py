from pathlib import Path
import os
import argparse
from subprocess import run
import configparser
import pdb

def run_nspe(f):
   #pdb.set_trace()
   config = configparser.ConfigParser()
   config.read(f)
   
   oospath  = config.get('Paths','oospath')
   #obspath  = config.get('Paths','obspath')
   #obsystem = config.get('Paths','obsystem')
                         
   #iniPath = Path(obspath,obsystem,'iniFiles')
   #inifiles = list(iniPath.glob('*Spr*.ini'))

   config = configparser.ConfigParser()
   config.read(f)

   
   peFilePath = config.get('Paths','pefilepath')
   #pdb.set_trace()
   os.chdir(peFilePath)
   cmd = 'c:/Users/500138/Documents/projects/oceans/pythonTools/run_nspe.sh'
   run(cmd)
   
   os.chdir(oospath)

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
