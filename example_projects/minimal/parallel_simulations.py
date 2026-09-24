# %% imports
import argparse
from multiprocessing import Pool

from susi.io.execution_config import SimulationParams, MultipleSusis
from susi.core.susi_main import Susi

# parameters.py beside this script: the project's parameters and its folder.
import parameters
from susi.io.metadata_model import SimulationMetaData
from susi.io.susi_parameter_model import SusiParams

# %% Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "n_parallel_processes", help="Number of parallel processes", type=int
)
cli_args = parser.parse_args()

# %% Create the scenarios


# This is the best way I found to create multiple parameter models based on one.
# There are some more here: https://github.com/pydantic/pydantic/discussions/3352
# First, define a function to be able to do this repeatedly
def create_depth_scenarios(
    base_params: SusiParams, ditch_depth_west: float
) -> SusiParams:
    """
    Example function to create new parameter models based on another one.
    This function changes the value of the parameter SusiParams.site_parameters.ditch_depth_west,
    and returns a fully validated model.
    """

    # 1. Get parameters of the base model into a Python dictionary
    data = base_params.model_dump(exclude_computed_fields=True)

    # 2. Modify the Python dictionary. Here we choose to change the L parameter
    data["site_parameters"]["ditch_depth_west"] = [ditch_depth_west]

    # 3. Validate the model to check that you did not make a mistake
    return SusiParams.model_validate(data)


# Next, create the scenarios
shallow = create_depth_scenarios(
    base_params=parameters.PARAMETERS, ditch_depth_west=-0.2
)
deep = create_depth_scenarios(base_params=parameters.PARAMETERS, ditch_depth_west=-0.7)


# Finally, create the list of parameters that will go into the susi simulation
all_parameters = [
    SimulationParams(
        metadata=SimulationMetaData(
            project_dir=parameters.PROJECT_DIR,
            run_id="run_01",
            stand_id="stand_01",
            scenario_id="deep_ditch",
        ),
        susi_params=deep,
    ),
    SimulationParams(
        metadata=SimulationMetaData(
            project_dir=parameters.PROJECT_DIR,
            run_id="run_01",
            stand_id="stand_01",
            scenario_id="shallow_ditch",
        ),
        susi_params=shallow,
    ),
]

execution_config = MultipleSusis(
    simulation_parameter_list=all_parameters,
    n_parallel_processes=cli_args.n_parallel_processes,
)


def run_susi(simulation_parameters: SimulationParams) -> None:
    # Initiate susi class
    susi = Susi(simulation_parameters)

    # Run simulation
    susi.run()

    # Save stuff
    susi.write_params_and_metadata()


# %% Execute parallel processing
if __name__ == "__main__":
    with Pool(processes=execution_config.n_parallel_processes) as pool:
        pool.map(func=run_susi, iterable=execution_config.simulation_parameter_list)
