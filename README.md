# Overview
Peatland simulator SUSI version used in Saari et al. (nimi) 2025 and Niemi et al. (2025) (nimi)

SUSI version was built in Python using Spyder programming tool. All packages and versions for running 
Spyder and this SUSI version can be found from requirements.txt

# Installation
`pip install -e .`

# Run
`python src/scripts/susi_calls.py`

# Project structure
SUSI project contains four folders:

inputs-folder 
- susi_para.py contains soil and site parameters and forest management recipies
- includes forest development (Motti) -files for five fertility-class sites with typical forest variables. 
   Development of stand allometry has been done for 50...60 years without management. Separate files are for 
    - Southern Finland (SF)
    - Central Finland (CF)
    - Northern Ostrobothnia - Kainuu area (NOBK)
    - Lapland (Lap)
  Fertility class has been denoted with the first number in the file name. e.g. Lap_22.xlsx. Fertility classes are:
    - 2: Ruohoturvekangas, Herb-rich, fertile
    - 3: Mustikkaturvekangas, Bilberry, Medium-fertile
    - 4: Puolukkaturvekangas, Lingonberry, Medium-poor
    - 5: Varputurvekangas, Dwarf shrub, Poor
  Dominant tree species is denoned by the second nuber in the file name: e.g.CF_31.xlsx
    - 1: Scots pine, Pinus sylvestris
    - 2: Norway spruce, Picea abies
  
- Weather files contain daily weather input data from Jan, 1, 2004 to Dec, 31, 2023.
    -	Separate weather files for SF, CF, NOBK and Lap

susi-folder 
- Includes all source codes for running the SUSI-simulations.
    - allometry.py  reads and processes Motti-files to produce allometric paths for stand development
    - canopygrid.py hydrology module for above-ground hydrology
    - canopylayer.py processes stem-wise development of trees
    - esom.py is organic matter decomposition model
    - fertilization.py processes release of nutrients from the fertilizers
    - figures.py   processes default visualization for the model runs
    - gvegetation processes groundvegetation development, mass and nutrient contets and litterfall
    - methane.py  applies simple methane balance model
    - mosslayer.py applies hydrological model for moss layer and surface flow along the model strip from ditch to ditch
    - outputs.py organizes outputs and locates them into netCDF4 files
    - stand.py integrates canopylayers and keeps track on stand-wise development 
    - susi_io.py operates some output and input procedures
    - susi_main.py is the main computation file
    - susi_utils.py operates some miscellaneous functions
    - temperature.py calculates vertical temperature profiles to peat.

tests-folder
    - susi_calls.py operates a single simulation: give weather files, 
     Motti-files, ditch depth, peat type, degree of decomposition,
     mor properties, forest harvesting, fertilization 
    - susi_calls_all.py runs scenarios with ditch shallowing or without shallowing 
    - fig reality check.py builds figures for stand growth and net soil C storage change
    - figures_ecosystem_services.py draws figures of differences between scenarios 
      and ecosystem services in scenarios

outputs-folder
    - running SUSI with susi_calls.py writes the outputs into outputs folder in susi.nc file
    - running SUSI with susi_calls_scenarios.py writes the outputs into outputs/shallow or outputs/no_shallow folders
