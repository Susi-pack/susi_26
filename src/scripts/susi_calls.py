# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:10:42 2020

@author: alauren
"""

from susi.io.execution_config import SimulationParams
from susi.core.susi_main import Susi
from inputs.parameters import golden_test
from susi.io.metadata_model import SimulationMetaData

# read weather input
simulation_parameters = SimulationParams(
    metadata=SimulationMetaData(experiment_id="your_experiment_name_here"),  #your_experiment_name_here
    susi_params=golden_test.PARAMETERS,
)

# Instantiate susi class
susi = Susi(
    simulation_parameters=simulation_parameters,
)

# Run simulation
susi.run()

# Save stuff
susi.write_params_and_metadata()
