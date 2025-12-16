# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:10:42 2020

@author: alauren
"""
import numpy as np
import datetime
from susi.susi_utils import read_FMI_weather
from inputs.susi_para import get_susi_para
from susi.susi_main import Susi

stands = [
          'SF_22.xlsx','SF_31.xlsx','SF_32.xlsx', 'SF_41.xlsx','SF_51.xlsx',   # Mottifile names in inputs folder
          'CF_22.xlsx','CF_31.xlsx','CF_32.xlsx', 'CF_41.xlsx','CF_51.xlsx',
          'NOBK_22.xlsx','NOBK_31.xlsx','NOBK_32.xlsx', 'NOBK_41.xlsx','NOBK_51.xlsx',
          'Lap_22.xlsx','Lap_31.xlsx','Lap_32.xlsx', 'Lap_41.xlsx','Lap_51.xlsx'
          ]

weather_inputs = ['SFw.csv','SFw.csv','SFw.csv','SFw.csv','SFw.csv',           # weather files in inputs folder 
                  'CFw.csv','CFw.csv','CFw.csv','CFw.csv','CFw.csv',
                  'NOBKw.csv','NOBKw.csv','NOBKw.csv','NOBKw.csv','NOBKw.csv',
                  'Lapw.csv','Lapw.csv','Lapw.csv','Lapw.csv','Lapw.csv'
                  ]
outnames = ['SF_22.nc','SF_31.nc','SF_32.nc','SF_41.nc','SF_51.nc',            # outputfiles to be written in outputs folder
            'CF_22.nc','CF_31.nc','CF_32.nc','CF_41.nc','CF_51.nc',
            'NOBK_22.nc','NOBK_31.nc','NOBK_32.nc','NOBK_41.nc','NOBK_51.nc',
            'Lap_22.nc','Lap_31.nc','Lap_32.nc','Lap_41.nc','Lap_51.nc',
            ]
sfcs = [2,3,3,4,5,                                                             # site fertility classes
        2,3,3,4,5,
        2,3,3,4,5,
        2,3,3,4,5
        ]

humus =[6.9, 5.8, 7.0, 5.5, 5.2,                                               # Mor layer mass after 50 yrs spinoff, kg/m2
        6.7, 5.6, 6.7, 5.3, 4.9,
        7.0, 5.4, 7.0, 5.5, 4.9,
        6.9, 5.9, 7.2, 6.0, 5.2
        ]
#humus = np.ones(len(humus))*2.0

for stand, weather_input,outname,sitetype,hum in zip(stands, weather_inputs,outnames, sfcs, humus):
    #***************** local call for SUSI*****************************************************

    #folderName=r'../outputs/no_shallow/'                                       # location where outputs are saved
    folderName=r'../outputs/shallow/'
                                                      
    
    wpath = r'../inputs/'                                                      # Folder where the weather files are located
      
    mottifile = {'path':r'../inputs/',                                         # Input file folder
                  'dominant':{1: stand},
                  'subdominant':{0:'susi_motti_input_lyr_1.xlsx'},
                  'under':{0:'susi_motti_input_lyr_2.xlsx'}} 
    
    wdata = weather_input                                                      # locate weather input file
    
    start_date = datetime.datetime(2004,1,1)                                   # start year of simulation
    end_date=datetime.datetime(2023,12,31)                                     # end year of simulation
    start_yr = start_date.year 
    end_yr = end_date.year
    yrs = (end_date - start_date).days/365.25
    
    sarkaSim = 40.                                                             # strip width, ie distance between ditches, m
    n = int(sarkaSim / 2)                                                      # number of computation nodes in the strip
    
    ageSim = {'dominant': 60.*np.ones(n),                                      # age of stand when starting the simulation, yrs, nodewise along the strip  
              'subdominant': 0*np.ones(n),
              'under': 0*np.ones(n)}                                                         
    
    sfc =  np.ones(n, dtype=int)*sitetype                                      # site fertility class
    
    #ageSim['dominant'][int(n/2):] = 2.
    #ageSim[4:-4] = 2.
    
    site = 'develop_scens'                                                     # parameter setup from get_susi_para
    
    forc=read_FMI_weather(0, start_date, end_date, sourcefile=wpath+wdata)     # read weather input
                
    wpara, cpara, org_para, spara, outpara, photopara = get_susi_para(wlocation='undefined', peat=site, 
                                                                              folderName=folderName, hdomSim=None,  
                                                                              ageSim=ageSim, sarkaSim=sarkaSim, sfc=sfc, 
                                                                              n=n)

    spara['drain_age'] =  100.
    mass_mor = hum 
    
    if np.median(sfc) > 4:
        spara['peat type']=['S','S','S','S','S','S','S','S']
        spara['peat type bottom']=['A']
        spara['vonP top'] =  [2,5,5,5,6,6,7,7] 
        spara['anisotropy'] = 10
        spara['rho_mor'] = 80.0
    else:
        spara['vonP top'] =  [2,5,5,5,6,6,7,7] 
        spara['anisotropy'] = 10
        spara['rho_mor'] = 90.0
        
    
    spara['h_mor'] = mass_mor/ spara['rho_mor'] 
    
    spara['ditch depth west'] = [-0.3, -0.6, -0.9]                             # ditch depth at the beginning of simulation m, if given several values SUSI calculates scenarios for each ditch depth  
    spara['ditch depth east'] = [-0.3, -0.6, -0.9]
    spara['ditch depth 20y west'] = [-0.2, -0.4, -0.6]                                           
    spara['ditch depth 20y east'] = [-0.2, -0.4, -0.6]                                           
    #spara['ditch depth 20y west'] = [-0.3, -0.6, -0.9]                                          
    #spara['ditch depth 20y east'] = [-0.3, -0.6, -0.9]                                          
    
    spara['scenario name'] =  ['D30', 'D60','D90']                             # scenario names
    
    outpara['netcdf'] = outname                                                # name the output files
    susi = Susi()                                                              # instance of susi class 
     
    susi.run_susi(forc, wpara, cpara, org_para, spara, outpara, photopara, start_yr, end_yr, wlocation = 'undefined', 
                                    mottifile=mottifile, peat= 'other', photosite='All data', 
                                    folderName=folderName,ageSim=ageSim, sarkaSim=sarkaSim, sfc=sfc)
        