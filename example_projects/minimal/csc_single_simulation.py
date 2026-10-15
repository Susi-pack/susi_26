# csc_parameters.py beside this script: the project's parameters and its folder.
import csc_parameters

from susi.core.susi_main import Susi
from susi.io.execution_config import SimulationParams
from susi.io.metadata_model import SimulationMetaData

# read weather input
simulation_parameters = SimulationParams(
    metadata=SimulationMetaData(
        project_dir=csc_parameters.PROJECT_DIR,
        run_id="run_single",  # your_run_name_here
    ),
    susi_params=csc_parameters.PARAMETERS,
)

# Instantiate susi class
susi = Susi(
    simulation_parameters=simulation_parameters,
)

# Run simulation
susi.run()

# Save stuff
susi.write_params_and_metadata()
