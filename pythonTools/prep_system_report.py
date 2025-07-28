# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 11:48:53 2024

@author: 500138
"""
import argparse

def prep_system_report(obssystem):
  """ create csv system report"""    
  print(obssystem)
  
def main():
    parser = argparse.ArgumentParser(description=
        'Input OOS system.\n')
    parser.add_argument(
        "obsystem", type=str, default=None,
        help="Enter the path to ini files. ex. ENAM")

    args    = parser.parse_args()
    obsystem =  args.obsystem
    #iniPath = Path(os.getcwd(),'case_studies',obsystem,'iniFiles')
    #pdb.set_trace()
    #prep_nspe(obsystem)
    prep_system_report(obsystem)

if __name__ == '__main__':
  main()