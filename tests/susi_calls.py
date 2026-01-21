# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:10:42 2020

@author: alauren
"""
# THIS files
import numpy as np
import datetime
from susi.susi_utils import read_FMI_weather
from inputs.susi_para import get_susi_para
from susi.susi_main import Susi

#***************** local call for SUSI*****************************************************
folderName=r'../outputs/'                                                      # location where outputs are saved

wpath = r'../inputs/'                                                          # Folder where the weather files are located
wdata='CFw.csv'                                                                # Weather file name

mottifile = {'path':r'../inputs/',                                             # Input file folder
              'dominant':{1: 'CF_41.xlsx'},                                    # Motti-file for the dominant layer 
              'subdominant':{0:'susi_motti_input_lyr_1.xlsx'},                 # subdominant layer Mottifle, 0 if not in use
              'under':{0:'susi_motti_input_lyr_2.xlsx'}}                       # understorey layer Mottifile, 0 if not in use


start_date = datetime.datetime(2004,1,1)                                       # Start date for simulation           
end_date=datetime.datetime(2007,12,31)                                         # End day for simulation
start_yr = start_date.year          
end_yr = end_date.year
yrs = (end_date - start_date).days/365.25

sarkaSim = 40.                                                                 # Strip width, ie distance between ditches, m
n = int(sarkaSim / 2)                                                          # Number of computation nodes in the strip, 2-m width of node

ageSim = {'dominant': 100.*np.ones(n),                                         # age of the stand in the beginning of the simulation, yrs, given for all nodes along the strip
          'subdominant': 0*np.ones(n),                                         # same for subdominant layer 
          'under': 0*np.ones(n)}                                               # same for understorey layer

site_fertility_class = 4
sfc =  np.ones(n, dtype=int) * site_fertility_class                            # site fertility class

site = 'develop_scens'                                                         # name of the parameter set in get_susi_para

forc=read_FMI_weather(0, start_date, end_date, sourcefile=wpath+wdata)         # read weather input
            
wpara, cpara, org_para, spara, outpara, photopara = get_susi_para(wlocation='undefined', peat=site, 
                                                                          folderName=folderName, hdomSim=None,  
                                                                          ageSim=ageSim, sarkaSim=sarkaSim, sfc=sfc, 
                                                                          n=n)

spara['cutting_yr'] = 2004                                                     # cutting year, not used if year is outside the simulation period
spara['drain_age'] =  100.                                                     # time since drainage, yrs
mass_mor = 1.616*np.log(spara['drain_age'])-1.409                              # Pitkänen et al. 2012 Forest Ecology and Management 284 (2012) 100–106

if np.median(sfc) > 4:
    spara['peat type']=['S','S','S','S','S','S','S','S']                       # Peat type 'S' if Sphagnum, 'A' if woody or Carex-peat    
    spara['peat type bottom']=['A']
    spara['vonP top'] =  [2,5,5,5,6,6,7,7]                                     # Degree of decomposition
    spara['anisotropy'] = 10                                                   # Anisotropy of peat hydraulic conductivity  
    spara['rho_mor'] = 80.0                                                    # bulk density of mor layer kg m-3 
else:
    spara['vonP top'] =  [2,5,5,5,6,6,7,7] 
    spara['anisotropy'] = 10
    spara['rho_mor'] = 90.0
    
spara['h_mor'] = mass_mor/ spara['rho_mor'] 

spara['ditch depth west'] = [-0.5]                                             # ditch depth at the beginning of simulation m, if given several values SUSI calculates scenarios for each ditch depth  
spara['ditch depth east'] = [-0.5]
spara['ditch depth 20y west'] = [-0.5]                                         # Ditch depth after 20 yrs, m, negative down
spara['ditch depth 20y east'] = [-0.5]                                         # Ditch depth after 20 yrs, m, negative down
spara['scenario name'] =  ['D60']                                              # Scanario names, equal nmber of names than ditch depth scenarios
#spara['enable_peatmiddle'] = False,
#spara['enable_peatbottom'] = False

susi = Susi()                                                                  # Initaiate susi class
 
susi.run_susi(forc, wpara, cpara, org_para, spara, outpara, photopara, start_yr, end_yr, wlocation = 'undefined', 
                                mottifile=mottifile, peat= 'other', photosite='All data', 
                                folderName=folderName,ageSim=ageSim, sarkaSim=sarkaSim, sfc=sfc)           #Run susi
    
#%%

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

from scipy import stats
from netCDF4 import Dataset  



folderName=r'../outputs/'


outname = 'susi.nc'

ff = folderName + outname

ncf=Dataset(ff, mode = 'r')                             # no shallowing

out1 = ncf['stand']['harvested_volume'][0, 1:, 1:-1]
out2 = ncf['stand']['harvested_log_volume'][0,1:, 1:-1]
out3 = ncf['stand']['harvested_pulp_volume'][0,:, 1:-1]
out4 = ncf['stand']['harvested_biomass'][0,:, 1:-1]
out5 = ncf['stand']['harvested_stems'][0,:, 1:-1]
out6 = ncf['stand']['mean_diameter'][0,1:, 1:-1]


print ('harvested volume')
print (out1)
print ('harvested log volume')
print (out2)
print ('harvested pulp volume')
print (out3)
print ('harvested biomass')
print (out4)
print ('harvested stems')
print (out5)
print ('mean diameter')
print (out6)

ncf.close()

