# -*- coding: utf-8 -*-

from susi.io.execution_config import SimulationParams
from susi.core.susi_main import Susi
from system_inputs.parameters import sample_parameters
from susi.io.metadata_model import SimulationMetaData
from susi.io.project_layout import project_dir

# read weather input
simulation_parameters = SimulationParams(
    metadata=SimulationMetaData(
        project_dir=project_dir("testing"),  # your_project_name_here
        run_id="testing2",  # your_run_name_here
    ),
    susi_params=sample_parameters.PARAMETERS,
)

# Instantiate susi class
susi = Susi(
    simulation_parameters=simulation_parameters,
)

# Run simulation
susi.run()

# Save stuff
susi.write_params_and_metadata()
