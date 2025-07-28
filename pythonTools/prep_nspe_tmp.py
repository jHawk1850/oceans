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
    pdb.set_trace()
    for f in inifiles:
        print(f)

def main():
    parser = argparse.ArgumentParser(description=
        'Input path to ini files.\n')
    parser.add_argument(
        "obsystem", type=str, default=None,
        help="Enter the path to ini files. ex. ENAM")

    args    = parser.parse_args()
    obsystem =  args.obsystem
    iniPath = Path(os.getcwd(),'case_studies',obsystem,'iniFiles')
    pdb.set_trace()
    #prep_nspe(obsystem)
    prep_nspe(iniPath)

if __name__ == '__main__':
  main()