# -*- coding: utf-8 -*-

from susi.io.execution_config import SimulationParams
from susi.core.susi_main import Susi

# parameters.py beside this script: the project's parameters and its folder.
import parameters
from susi.io.metadata_model import SimulationMetaData

# read weather input
simulation_parameters = SimulationParams(
    metadata=SimulationMetaData(
        project_dir=parameters.PROJECT_DIR,
        run_id="testing2",  # your_run_name_here
    ),
    susi_params=parameters.PARAMETERS,
)

# Instantiate susi class
susi = Susi(
    simulation_parameters=simulation_parameters,
)

# Run simulation
susi.run()

# Save stuff
susi.write_params_and_metadata()
